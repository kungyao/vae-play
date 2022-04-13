from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn

from torch.nn.functional import grid_sample
from torchvision.models.resnet import resnet50
from torchvision.models.densenet import densenet121

try:
    from models.blocks import *
except:
    from blocks import *

SAMPLE_SCALE = 2
SAMPLE_COUNT = int(360 * SAMPLE_SCALE)
VALUE_WEIGHT = 10

class ContentEndoer(nn.Module):
    def __init__(self):
        super().__init__()
        # resnet = resnet50(pretrained=True)
        # self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # backbone = resnet_fpn_backbone('resnet50', True)
        self.out_channels = 256
        self.convs = nn.Sequential(
            Conv2d(3, 32, 5), 
            Conv2d(32, 64, 5, stride=2), 
            Conv2d(64, 128, 5, stride=2), 
            # deepcopy(resnet.layer1), # 256
            # deepcopy(resnet.layer2), # 512
            # deepcopy(resnet.layer3), # 1024
            # deepcopy(resnet.layer4), # 2048
            Conv2d(128, self.out_channels, 3, stride=2),  
            Conv2d(self.out_channels, self.out_channels, 3, stride=2),
            Conv2d(self.out_channels, self.out_channels, 3),
            Conv2d(self.out_channels, self.out_channels, 3)
        )

    def forward(self, x: torch.Tensor):
        x = self.convs(x)
        return x

class EllipseParamPredictor(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.convs = nn.Sequential(
            Conv2d(in_channels, in_channels, 3, stride=2, bn=None, activate="lrelu"), 
            Conv2d(in_channels, in_channels, 3, stride=2, bn=None, activate="lrelu"), 
            Conv2d(in_channels, in_channels, 3, stride=2, bn=None, activate="lrelu"), 
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fcs = nn.Sequential(
            Linear(in_channels, in_channels, activate=None), 
            Linear(in_channels, in_channels, activate=None), 
            # For cx, cy, rx, ry, step.
            Linear(in_channels, 5, activate=None), 
        )
    
    def forward(self, x: torch.Tensor):
        # x = self.convs(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fcs(x)
        return x

class ValueEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, fix_steps=SAMPLE_COUNT):
        super().__init__()
        self.out_channels = 128
        self.fcs = nn.Sequential(
            Linear(in_channels, 64, activate=None), 
            Linear(64, 128, activate=None), 
            Linear(128, 256, activate=None), 
            Linear(256, out_channels, activate=None)
        )
        self.attns = nn.Sequential(
            SelfAttentionBlock(fix_steps), 
            SelfAttentionBlock(fix_steps), 
            SelfAttentionBlock(fix_steps)
        )

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)
        x = x.reshape(b * c, h * w)
        # from embeded channel to output channel.
        x = self.fcs(x)
        x = x.reshape(b, c, -1, w)
        # attention between point ans point.
        x = self.attns(x)
        return x  

class EmitLineParamPredictor(nn.Module):
    def __init__(self, fix_steps=SAMPLE_COUNT, in_channels=256):
        super().__init__()
        self.embed_size = 5 + 3
        self.value_encoder = ValueEncoder(self.embed_size, in_channels, fix_steps=fix_steps)

        self.batch_attention_a = nn.Sequential(
            SelfAttentionBlock(fix_steps), 
            SelfAttentionBlock(fix_steps), 
            SelfAttentionBlock(fix_steps)
        )
        self.trigger_pred = nn.Sequential(
            Linear(in_channels, in_channels, activate='lrelu'), 
            Linear(in_channels, in_channels, activate='lrelu'), 
            Linear(in_channels, 2, activate=None)
        )

        self.batch_attention_b = nn.Sequential(
            SelfAttentionBlock(fix_steps), 
            SelfAttentionBlock(fix_steps), 
            SelfAttentionBlock(fix_steps)
        )
        # -10~10
        # -10~10
        # -1~1
        # 0~20
        # offset_x, offset_y, theta, length
        self.params_pred = nn.Sequential(
            Linear(in_channels, in_channels, activate='lrelu'), 
            Linear(in_channels, in_channels, activate=None), 
            Linear(in_channels, 4, activate=None)
        )
    
    def forward(self, x: torch.Tensor, sample_infos: map, params: torch.Tensor):
        x = x.reshape(x.size(0), x.size(1), x.size(2), 1)
        # cx, cy, rx, ry
        param_embed = params[:, :4].reshape(x.size(0), 1, -1, 1).repeat(1, x.size(1), 1, 1).to(x.device)
        d_embed = (torch.remainder(torch.arange(0, x.size(1), 1).reshape(1, -1).repeat(x.size(0), 1), torch.round(params[:, 4]).reshape(-1, 1)) == 0).to(dtype=x.dtype, device=x.device)
        d_embed = d_embed.reshape(x.size(0), x.size(1), 1, 1)
        info_pts = torch.stack(sample_infos["sample"], dim=0)
        known_line_param_embeded = torch.cat(
            # dpx, dpy, radian
            [info_pts[:, :, 2], info_pts[:, :, 3], info_pts[:, :, 5]], 
            dim=-1) # (B * PT * 3)
        known_line_param_embeded = known_line_param_embeded.reshape(x.size(0), x.size(1), -1, 1).to(x.device)
        # x = torch.cat([x, params, known_line_param_embeded], dim=2)
        known_line_param_embeded = torch.cat([param_embed, d_embed, known_line_param_embeded], dim=2)
        known_line_param_embeded = self.value_encoder(known_line_param_embeded)

        x = x + known_line_param_embeded

        x_a = self.batch_attention_a(x)
        x_a = x_a.reshape(x_a.size(0) * x_a.size(1), x_a.size(2))
        if_trigger = self.trigger_pred(x_a)

        x_b = self.batch_attention_b(x)
        x_b = x_b.reshape(x_b.size(0) * x_b.size(1), x_b.size(2))
        preds = self.params_pred(x_b)
        return if_trigger, preds

def sample_points_ellipse(cx, cy, rx, ry, step, image_size):
    # step = int(step)
    # if step < 1 / SAMPLE_SCALE:
    #     step = 1 / SAMPLE_SCALE
    ds = torch.arange(0, SAMPLE_COUNT, 1)
    radians = ds / SAMPLE_SCALE * np.pi / 180
    pxs = cx + rx * torch.cos(radians)
    pys = cy + ry * torch.sin(radians)
    dpxs = rx * -torch.sin(radians)
    dpys = ry * torch.cos(radians)
    ldps = torch.sqrt(torch.square(dpxs) + torch.square(dpys))
    dpxs /= ldps
    dpys /= ldps
    rot_fix = torch.tensor(-np.pi / 2)
    tmp_dpxs = dpxs * torch.cos(rot_fix) - dpys * torch.sin(rot_fix)
    tmp_dpys = dpxs * torch.sin(rot_fix) + dpys * torch.cos(rot_fix)
    dpxs = tmp_dpxs
    dpys = tmp_dpys
    samples = torch.stack([pxs, pys, dpxs, dpys, ds, radians], dim=-1)
    samples = samples.to(dtype=torch.float)
    return samples

class EmitLinePredictor(nn.Module):
    def __init__(self, image_size, in_channels=256):
        super().__init__()
        self.image_size = image_size
        self.convs = nn.Sequential(
            Conv2d(in_channels, 64, 3, stride=2, bn=None, activate="lrelu"), 
            Conv2d(64, 128, 3, stride=2, bn=None, activate="lrelu"), 
            Conv2d(128, 256, 3, stride=2, bn=None, activate="lrelu"), 
            Conv2d(256, 512, 3, stride=2, bn=None, activate="lrelu"), 
            Conv2d(512, 1024, 3, stride=2, bn=None, activate="lrelu"), 
            Conv2d(1024, 2048, 3, stride=1, bn=None, activate="lrelu"), 
            Conv2d(2048, 2048, 3, stride=1, bn=None, activate="lrelu"), 
        )

        # self.param_predictor = EmitLineParamPredictor(fix_steps=SAMPLE_COUNT, in_channels=in_channels)
        self.param_predictor = EmitLineParamPredictor(fix_steps=SAMPLE_COUNT, in_channels=2048)

    def process(self, x: torch.Tensor, params: torch.Tensor):
        b, c, h, w = x.shape
        w_half = (w - 1) / 2
        h_half = (h - 1) / 2
        sample_infos = {
            "size": [], 
            "sample": [], 
        }
        feature_points = []
        for i, (cx, cy, rx, ry, step) in enumerate(params):
        # for i, (cx, cy, rx, ry) in enumerate(params):
            # print(cx, cy, rx, ry, step)
            info_pts = sample_points_ellipse(cx, cy, rx, ry, 1, self.image_size)
            # 
            ellipse_ptx = info_pts[:, 0]
            ellipse_pty = info_pts[:, 1]
            # ellipse_ptx = (ellipse_ptx - w_half) / w_half
            # ellipse_pty = (ellipse_pty - h_half) / h_half
            # N * 2
            ellipse_pts = torch.stack([ellipse_ptx, ellipse_pty], dim=1)
            # 1 * 1 * N * 2
            ellipse_pts = ellipse_pts.unsqueeze(0).unsqueeze(0)
            ellipse_pts = ellipse_pts.to(x.device)
            # 1 * C * H * W
            # 'bilinear' | 'nearest' | 'bicubic'
            sample_pts = grid_sample(x[i][None], ellipse_pts, mode='bilinear')
            # 1 * C * 1 * N
            sample_pts = sample_pts.squeeze()
            # N * C
            sample_pts = sample_pts.permute(1, 0)
            # 
            sample_infos["size"].append(sample_pts.size(0))
            sample_infos["sample"].append(info_pts)
            feature_points.append(sample_pts)
        feature_points = torch.stack(feature_points, dim=0)
        return feature_points, sample_infos
    
    def forward(self, x: torch.Tensor, params: torch.Tensor):
        x = self.convs(x)
        # Weight
        params[:, :4] = params[:, :4] / VALUE_WEIGHT
        # Sample points on feature map. (batch, pt, in_channel(256)).
        feature_pts, sample_infos = self.process(x, params) 
        # Do predict.
        if_triggers, line_params = self.param_predictor(feature_pts, sample_infos, params) 
        if_triggers = if_triggers.split(sample_infos["size"], 0)
        line_params = line_params.split(sample_infos["size"], 0)
        return if_triggers, line_params, sample_infos

class ComposeNet(nn.Module):
    def __init__(self, image_size):
        super(ComposeNet, self).__init__()
        # self.add_coord = AddCoords(if_normalize=True)
        self.encoder = ContentEndoer()
        self.ellipse_predictor = EllipseParamPredictor(self.encoder.out_channels)
        # self.emit_line_predictor = EmitLinePredictor(image_size, in_channels=self.encoder.out_channels)
        self.emit_line_predictor = EmitLinePredictor(image_size, in_channels=3)

    def forward(self, x_base, x_attr):
        # x = self.add_coord(x)
        # x = self.encoder(x)
        # ellipse_params = self.ellipse_predictor(x)
        ellipse_params = self.ellipse_predictor(self.encoder(x_base))
        if_triggers, line_params, sample_infos = self.emit_line_predictor(x_attr, ellipse_params.detach().cpu())
        output = {}
        output.update(ellipse_params=ellipse_params)
        output.update(if_triggers=if_triggers)
        output.update(line_params=line_params)
        output.update(sample_infos=sample_infos)
        return output

if __name__ == "__main__":
    image_size = 512
    batch_size = 4

    net = ComposeNet(image_size)
    net.cuda()

    fake_img = torch.rand(batch_size, 3, image_size, image_size)
    fake_img = fake_img.cuda()

    # print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    # net.train()
    # res = net(fake_img)
    # ellipse_params = res["ellipse_params"]
    # if_triggers = res["if_triggers"]
    # line_params = res["line_params"]
    # sample_infos = res["sample_infos"]
    # print(ellipse_params.shape)
    # for i in range(batch_size):
    #     print(if_triggers[i].shape)
    #     print(line_params[i].shape)
    #     # print(sample_infos[i])

    print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
    net.eval()
    with torch.no_grad():
        res = net(fake_img)
        ellipse_params = res["ellipse_params"]
        if_triggers = res["if_triggers"]
        line_params = res["line_params"]
        for i in range(batch_size):
            print(ellipse_params[i].shape)
            print(if_triggers[i].shape)
            print(line_params[i].shape)
