from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import grid_sample
from torchvision.models.resnet import resnet18, resnet34

try:
    from models.blocks import *
except:
    from blocks import *
from tools.ops import initialize_model
from tools.utils import find_contour, resample_points

VALUE_WEIGHT = 10

class TMPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, if_down, bn=None):
        super().__init__()
        stride1 = 2 if if_down else 1
        self.convs = nn.Sequential(
            Conv2d(in_channels, out_channels, 3, stride=stride1, bn=bn, activate="lrelu"), 
            Conv2d(out_channels, out_channels, 1, stride=1, bn=None, activate="lrelu"), 
            Conv2d(out_channels, out_channels, 3, stride=1, bn=bn, activate="lrelu"), 
        )
        # self.skip = Conv2d(in_channels, out_channels, 3, stride=stride1, bn=None, activate="lrelu")
        # self.cat = Conv2d(out_channels * 2, out_channels, 3, stride=1, bn=None, activate="lrelu")

    def forward(self, x):
        x = self.convs(x)
        # x_skip = self.skip(x)
        # x = torch.cat([x_dir, x_skip], dim=1)
        # x = self.cat(x)
        return x

class ContentEndoer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.convs1 = nn.Sequential(
            TMPBlock(in_channels, 64, True, bn=None), 
            TMPBlock(64, 64, True, bn=None), 
            TMPBlock(64, 64, False, bn=None), 
            TMPBlock(64, 64, False, bn=None), 
            TMPBlock(64, 64, False, bn=None), 
            TMPBlock(64, 64, False, bn=None), 
            TMPBlock(64, 64, False, bn=None), 
            TMPBlock(64, 64, False, bn=None), 
        )
        self.convs2 = nn.Sequential(
            TMPBlock(in_channels, 64, True, bn="instance"), 
            TMPBlock(64, 64, True, bn="instance"), 
            TMPBlock(64, 64, False, bn="instance"), 
            TMPBlock(64, 64, False, bn="instance"), 
            TMPBlock(64, 64, False, bn="instance"), 
            TMPBlock(64, 64, False, bn="instance"), 
            TMPBlock(64, 64, False, bn="instance"), 
            TMPBlock(64, 64, False, bn="instance"), 
        )

        self.out_size = 128
        self.out_channels = 128

    def forward(self, x: torch.Tensor):
        x_1 = self.convs1(x)
        x_2 = self.convs2(x)
        x = torch.cat([x_1, x_2], dim=1)
        return x

class ValueEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, pt_size=4096):
        super().__init__()
        self.out_channels = 128
        self.fcs = nn.Sequential(
            Linear(in_channels, 64, activate=None), 
            Linear(64, 128, activate=None), 
            Linear(128, 256, activate=None), 
            Linear(256, out_channels, activate=None)
        )
        self.attns = nn.Sequential(
            SelfAttentionBlock(pt_size), 
            SelfAttentionBlock(pt_size), 
            SelfAttentionBlock(pt_size)
        )

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)
        x = x.reshape(b * c, h * w)
        # from embeded channel to output channel.
        x = self.fcs(x)
        x = x.reshape(b, c, -1, 1)
        # attention between point ans point.
        x = self.attns(x)
        return x  

class LinePredictor(nn.Module):
    def __init__(self, image_size, pt_size=4096, in_channels=256):
        super().__init__()
        self.max_point = pt_size
        
        # 
        self.frequency_encode_img = []
        level = int(np.log(image_size)) - 1
        tmp_channel = in_channels
        tmp_out_channel = min(in_channels, tmp_channel * 2)
        for _ in range(level):
            self.frequency_encode_img.append(Conv2d(tmp_channel, tmp_out_channel, 3, stride=2, bn="instance", activate="lrelu"))
            tmp_channel = tmp_out_channel
            tmp_out_channel = min(in_channels, tmp_channel * 2)
        tmp_out_channel = in_channels
        self.frequency_encode_img.append(Conv2d(tmp_channel, tmp_out_channel, 1, stride=1, bn=None, activate="lrelu"))
        self.frequency_encode_img.append(nn.AdaptiveAvgPool2d(1))
        self.frequency_encode_img = nn.Sequential(*self.frequency_encode_img)
        self.frequency_encode_img_sub =  nn.Sequential(
            Linear(tmp_out_channel, tmp_out_channel, activate='lrelu'), 
            Linear(tmp_out_channel, tmp_out_channel, activate=None), 
            Linear(tmp_out_channel, tmp_out_channel, activate=None)
        )
        
        #
        in_channels = in_channels * (1 + 1) + 2 + 2 # Plus embedding
        # self.batch_attention = nn.Sequential(
        #     SelfAttentionBlock(pt_size), 
        #     SelfAttentionBlock(pt_size), 
        #     SelfAttentionBlock(pt_size)
        # )

        self.frequency_head = nn.Sequential(
            Linear(in_channels, in_channels, activate='lrelu'), 
            Linear(in_channels, in_channels, activate='lrelu')
        )

        self.frequency_pred = nn.Sequential(
            Linear(in_channels, in_channels, activate='lrelu'), 
            Linear(in_channels, in_channels, activate='lrelu'), 
            Linear(in_channels, 1, activate=None),
            nn.Sigmoid()
        )

        # -10~10
        # -10~10
        # offset_x, offset_y
        self.params_pred = nn.Sequential(
            Linear(in_channels * 2, in_channels * 2, activate='lrelu'), 
            Linear(in_channels * 2, in_channels, activate='lrelu'), 
            Linear(in_channels, 2, activate=None)
        )

    def process(self, x: torch.Tensor, contours):
        dtype = x.dtype
        device = x.device
        b, c, h, w = x.shape
        resamples = []
        stacked_cnts = []
        for i, cnt in enumerate(contours):
            normalized_pts = cnt.to(device)
            stacked_cnt = torch.zeros(self.max_point, 2, dtype=dtype, device=device)
            if normalized_pts.numel() != 0:
                stacked_cnt[:normalized_pts.size(0)] = normalized_pts
                normalized_pts = normalized_pts.unsqueeze(0).unsqueeze(0)
                resample = grid_sample(x[i][None], normalized_pts, mode='bilinear')
                resample = resample.squeeze()
                resample = resample.permute(1, 0)
                if self.max_point > resample.size(0):
                    resample_padding = torch.zeros(self.max_point - resample.size(0), c, dtype=dtype, device=device)
                    resample = torch.cat([resample, resample_padding], dim=0)
            else:
                resample = torch.zeros(self.max_point, c, dtype=dtype, device=device)
            resamples.append(resample)
            stacked_cnts.append(stacked_cnt)
        resamples = torch.stack(resamples, dim=0)
        stacked_cnts = torch.stack(stacked_cnts, dim=0)
        return resamples, stacked_cnts
    
    def forward(self, x: torch.Tensor, contours, x_cls: torch.Tensor):
        b, c, h, w = x.shape
        # Sample points on feature map. (batch, pt, in_channel(256)).
        x_pt_feature, x_pt_cnts = self.process(x, contours) 
        # (batch, pt)
        x_freq_img = self.frequency_encode_img(x)
        x_freq_img = x_freq_img.reshape(x_freq_img.size(0), -1)
        x_freq_img = self.frequency_encode_img_sub(x_freq_img)

        # Do predict.
        x_cls = x_cls.softmax(dim=-1)
        x_pt_feature = x_pt_feature.reshape(b, self.max_point, c, 1)
        # 
        x = torch.cat([
            x_pt_feature, 
            x_pt_cnts.reshape(b, self.max_point, -1, 1), 
            x_freq_img.reshape(b, 1, c, 1).repeat(1, self.max_point, 1, 1), 
            x_cls.reshape(b, 1, -1, 1).repeat(1, self.max_point, 1, 1)
            ], dim=2
        )
        #  + self.batch_attention_2(x_pt_feature * x_freq_img.reshape(b, self.max_point, 1, 1).repeat(1, 1, c, 1))
        # x = self.batch_attention(x)
        # x = torch.cat([
        #     x, 
        #     x_pt_feature, 
        #     x_cls.reshape(b, 1, -1, 1).repeat(1, self.max_point, 1, 1)
        # ], dim=2)
        # x = x + x * x_freq_img.reshape(b, self.max_point, 1, 1).repeat(1, 1, c, 1)
        # print(x.shape)
        x = x.reshape(b, self.max_point, -1)
        # print(x.shape)
        size_per_img = [len(x) for x in contours]
        tmp_x = []
        for i in range(b):
            tmp_x.append(x[i][:size_per_img[i]])
        x = torch.cat(tmp_x, dim=0)
        # print(x.shape)
        # 
        x_freq = self.frequency_head(x)
        x = torch.cat([x, x_freq], dim=1)
        # x = x * x_freq
        x_pred = self.params_pred(x) 
        x_freq = self.frequency_pred(x_freq).squeeze()
        return x_pred, x_freq

class ClassPredictor(nn.Module):
    def __init__(self, in_size, in_channels, num_of_classes):
        super().__init__()
        max_channels = 2048

        self.convs = []
        in_channels = in_channels
        out_channels = min(in_channels * 2, max_channels)
        repeat_num = int(np.log2(in_size)) - 1
        for _ in range(repeat_num):
            # self.convs.append(nn.Sequential(
            #     Conv2d(in_channels, out_channels, 3, stride=2, bn="instance"), 
            #     Conv2d(out_channels, out_channels, 3, stride=1, bn="instance")
            # ))
            self.convs.append(Conv2d(in_channels, out_channels, 3, stride=2))
            in_channels = out_channels
            out_channels = min(in_channels * 2, max_channels)
        self.convs.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.convs = nn.Sequential(*self.convs)
        
        in_size = in_channels * 1 * 1
        self.cls_convs = nn.Sequential(
            Linear(in_size, in_size // 2, activate="lrelu"), 
            Linear(in_size // 2, in_size // 4, activate="lrelu"), 
            Linear(in_size // 4, num_of_classes, activate=None)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.reshape(x.size(0), -1)
        x = self.cls_convs(x)
        return x

class ComposeNet(nn.Module):
    def __init__(self, image_size, pt_size=4096):
        super(ComposeNet, self).__init__()
        self.max_point = pt_size

        self.add_coord = AddCoords(if_normalize=True)
        self.encoder = ContentEndoer(3 + 2)
        # 實心or射線
        # self.cls_classifier = nn.Sequential(
        #     resnet18(pretrained=True), 
        #     nn.Linear(1000, 2)
        # )
        self.cls_classifier = ClassPredictor(self.encoder.out_size, self.encoder.out_channels, 2)
        # 
        self.line_predictor = LinePredictor(self.encoder.out_size, pt_size=pt_size, in_channels=self.encoder.out_channels)

    def forward(self, x, target=None):
        if self.training and target is not None:
            size = []
            contours = []
            for t in target:
                size.append(len(t["points"]))
                contours.append(t["points"][:, :2])
            # contours = torch.stack(contours, dim=0)
        else:
            device = x.device
            b, c, h, w = x.shape
            size = []
            contours = []
            for i in range(b):
                cnt = find_contour(np.asarray(x[i][1].cpu()))
                cnt = resample_points(cnt, self.max_point)
                cnt = (cnt / h - 0.5) / 0.5
                cnt = torch.FloatTensor(cnt)
                size.append(len(cnt))
                contours.append(cnt.to(device=device))
            # contours = torch.stack(contours, dim=0)
        # 
        x = self.add_coord(x)
        x = self.encoder(x)
        # 
        x_cls = self.cls_classifier(x)
        # 
        x, x_freq = self.line_predictor(x, contours, x_cls.detach())
        x = x.split(size)
        x_freq = x_freq.split(size)
        return {
            "classes": x_cls, 
            "contours": contours, 
            "target_pts": x, 
            "target_frequency": x_freq, 
        }

class Discriminator(nn.Module):
    def __init__(self, image_size, pt_size=4096):
        super().__init__()
        self.max_point = pt_size
        # sx, sy, ex, ey
        self.target_sub_dim = 4
        max_channels = 512

        self.global_convs = []
        level = int(np.log2(image_size)) - 2 - 1
        in_channel = 32
        self.global_convs.append(Conv2d(3, in_channel, 3, stride=2, bn=None, activate="lrelu"))
        out_channel = min(in_channel * 2, max_channels)
        for _ in range(level - 1):
            self.global_convs.append(Conv2d(in_channel, out_channel, 3, stride=2, bn="instance", activate="lrelu"))
            in_channel = out_channel
            out_channel = min(in_channel * 2, max_channels)
        self.global_convs.append(Conv2d(in_channel, max_channels, 1, stride=1, bn=None, activate="lrelu"))
        self.global_convs.append(nn.AdaptiveAvgPool2d(1))
        self.global_convs = nn.Sequential(*self.global_convs)

        self.local_convs = []
        in_channel = pt_size * self.target_sub_dim
        out_channel = min(in_channel // 2, max_channels)
        for _ in range(level):
            self.local_convs.append(Linear(in_channel, in_channel, bias=None, activate='tanh'))
            self.local_convs.append(Linear(in_channel, out_channel, bias=None, activate=None))
            in_channel = out_channel
            out_channel = min(in_channel // 2, max_channels)
        self.local_convs.append(Linear(in_channel, max_channels, bias=None, activate='lrelu'))
        self.local_convs = nn.Sequential(*self.local_convs)

        self.merge_convs = nn.Sequential(
            Linear(max_channels * 2, max_channels * 2, bias=True, activate='lrelu'), 
            Linear(max_channels * 2, max_channels, bias=True, activate='lrelu'), 
            Linear(max_channels, max_channels, bias=True, activate='lrelu'), 
            Linear(max_channels, max_channels // 2, bias=True, activate='lrelu'), 
            Linear(max_channels // 2, 1, bias=False, activate=None), 
            nn.Sigmoid()
        )

    def process_target(self, target:torch.Tensor):
        if target.size(0) < self.max_point:
            padding = torch.zeros(self.max_point - target.size(0), self.target_sub_dim, dtype=target.dtype, device=target.device)
            target = torch.cat([target, padding], 0)
        return target

    def forward(self, imgs, targets):
        # Assert that target.size(0) also means point size is smaller than self.max_point.
        targets = torch.stack([self.process_target(t) for t in targets], dim=0)
        targets = targets.reshape(targets.size(0), -1)
        # 
        global_feat = self.global_convs(imgs).squeeze()
        local_feat = self.local_convs(targets)
        # 
        merge_feat = torch.cat([global_feat, local_feat], dim=1)
        out = self.merge_convs(merge_feat).squeeze()
        return out
