from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.functional import grid_sample
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from tools.utils import find_contour, resample_points
try:
    from models.blocks import *
except:
    from blocks import *

# 
# TODO: 
# Test MAX_POINTS for optimization
# avg_count 910
# max_count 2563
# max_rdp_count 229
# 

CASE = 1
DEFAULT_MAX_POINTS = 256
def find_tensor_contour(x: torch.Tensor, max_points: int=DEFAULT_MAX_POINTS, threshold: float=0.5):
    contours = []
    for mm in x:
        tmp_m = mm.detach().cpu()
        tmp_m = tmp_m.squeeze().numpy().copy()
        tmp_m[tmp_m>=threshold] = 1
        tmp_m[tmp_m<threshold] = 0
        contour = find_contour(tmp_m)
        contour = resample_points(contour, max_points=max_points)
        contours.append(torch.tensor(contour, dtype=x.dtype))
    return contours

# B * MAX_POINTS * H * W
def make_embeding_tensor(contours: List, img_size: torch.Size, max_points: int=DEFAULT_MAX_POINTS):
    # Make tensor
    embedings = []
    h, w = img_size
    for cnt in contours:
        # pt_size = cnt.size(0)
        # pad_size = max_points - pt_size
        embeding_c = torch.zeros(max_points, h, w, dtype=cnt.dtype)
        pts = torch.cat([torch.arange(cnt.size(0), dtype=cnt.dtype).view(-1, 1), cnt], dim=-1)
        pts = pts.to(dtype=torch.long)
        embeding_c[pts[:, 0], pts[:, 2], pts[:, 1]] = 1
        embedings.append(embeding_c)
    embedings = torch.stack(embedings, dim=0)
    return embedings

# B * MAX_POINTS * C
def resample_feature(feature, contours, max_points: int=DEFAULT_MAX_POINTS):
    dtype = feature.dtype
    device = feature.device
    b, c, h, w = feature.shape
    w_half = (w - 1) / 2
    h_half = (h - 1) / 2
    resamples = []
    for i, cnt in enumerate(contours):
        normalized_pts = cnt.to(device)
        if normalized_pts.numel() != 0:
            normalized_pts[:, 0] = (normalized_pts[:, 0] - w_half) / w_half
            normalized_pts[:, 1] = (normalized_pts[:, 1] - h_half) / h_half
            normalized_pts = normalized_pts.unsqueeze(0).unsqueeze(0)
            resample = grid_sample(feature[i][None], normalized_pts, mode='bicubic')
            resample = resample.squeeze()
            resample = resample.permute(1, 0)
            if max_points > resample.size(0):
                resample_padding = torch.zeros(max_points - resample.size(0), c, dtype=dtype, device=device)
                resample = torch.cat([resample, resample_padding], dim=0)
        else:
            resample = torch.zeros(max_points, c, dtype=dtype, device=device)
        resamples.append(resample)
    resamples = torch.stack(resamples, dim=0)
    return resamples

class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.feature = resnet_fpn_backbone('resnet50', True)
        self.out_channels = self.feature.out_channels

    def forward(self, x):
        # 0 : size / 4
        # 1 : size / 8
        # 2 : size / 16
        # 3 : size / 32
        # pool : size / 64
        x = self.feature(x)
        return x["0"]

class MaskNet(nn.Module):
    def __init__(self, in_channel):
        super(MaskNet, self).__init__()
        # 
        self.conv1 = nn.Sequential(
            Conv2d(in_channel, in_channel // 2, 3, stride=1), 
            Conv2d(in_channel // 2, in_channel // 4, 3, stride=1),
            Conv2d(in_channel // 4, in_channel // 8, 3, stride=1)
        )
        in_channel = in_channel // 8
        # self.attn1 = SCSEBlock(in_channel, reduction=4)
        # 
        self.conv2 = nn.Sequential(
            Conv2d(in_channel, in_channel // 2, 3, stride=1), 
            Conv2d(in_channel // 2, in_channel // 4, 3, stride=1)
        )
        in_channel = in_channel // 4
        # self.attn2 = SCSEBlock(in_channel, reduction=4)
        # 
        self.predictor = nn.Sequential(
            Conv2d(in_channel, in_channel // 2, 3, stride=1, bn=False, activate=None), 
            Conv2d(in_channel // 2, 1, 3, stride=1, bn=False, activate=None)
        )

    def forward(self, x):
        x = self.conv1(x)
        # x = self.attn1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv2(x)
        # x = self.attn2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        # x = self.attn3(x)
        x = self.predictor(x)
        return x

class RefineNet(nn.Module):
    def __init__(self, in_channel, in_size):
        super(RefineNet, self).__init__()

        self.deform_blocks = []
        #################
        iters = 6
        for _ in range(iters):
            self.deform_blocks.append(SelfAttentionBlock(in_channel))
        self.deform_blocks = nn.ModuleList(self.deform_blocks)
        fc_in_size = in_channel*in_size*1
        self.fc_blocks = nn.Sequential(
            nn.Linear(fc_in_size, fc_in_size//8), 
            nn.Linear(fc_in_size//8, in_channel * 2)
        )

    def forward(self, x):
        if CASE == 1:
            b, c, hw = x.shape
            x = x.unsqueeze(-1)
            for m in self.deform_blocks:
                x = m(x)
            x = x.view(b, -1)
            x = self.fc_blocks(x)
            x = x.view(b, c, 2)
        elif CASE == 2:
            pass
        return x

class ComposeNet(nn.Module):
    def __init__(self, padding:int =1, max_points=DEFAULT_MAX_POINTS):
        super(ComposeNet, self).__init__()
        # Feature extract
        self.feature_net = FeatureNet()
        # Generate initial mask region
        self.mask_net = MaskNet(self.feature_net.out_channels)
        # Refine contour point on mask region. Plus 2 for coords.
        self.refine_net = RefineNet(max_points, self.feature_net.out_channels+2)
        # Expand two new channel for coordinate
        self.add_coords = AddCoords()

        self.max_points = max_points
        self.padding_for_contour = padding
        # Initialize parameter
        self.initialize(self.mask_net)
        self.initialize(self.refine_net)

    def initialize(self, mm: nn.Module):
        for m in mm.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        device = x.device
        padding = self.padding_for_contour
        feature = self.feature_net(x)

        mask_out = self.mask_net(feature)
        # B * [N * 2]. (N may be different)
        contours = find_tensor_contour(F.pad(mask_out.sigmoid(), (padding, padding, padding, padding), "constant", 0), max_points=self.max_points)
        if CASE == 1:
            feature = F.pad(feature, (padding, padding, padding, padding), "constant", 0)
            feature = self.add_coords(feature)
            # B * MAX_POINTS * C
            feature_embed = resample_feature(feature, contours, max_points=self.max_points)
        elif CASE == 2:
            # B * T(MAX_POINTS) * Hc * Wc
            # Let non-contour-point patch be zero.
            contours_embeding = make_embeding_tensor(contours, feature.shape[-2:], max_points=self.max_points)
            contours_embeding = contours_embeding.to(device)
            # Scale to original size
            feature = F.interpolate(feature, scale_factor=4, mode='bilinear')
            feature = F.pad(feature, (padding, padding, padding, padding), "constant", 0)
            # B * (C + MAX_POINTS + 2) * Hc * Wc
            feature_embed = torch.cat([self.add_coords(feature), contours_embeding], dim=1)
        # B * MAX_POINTS * 2
        contour_regressions = self.refine_net(feature_embed)
        
        return {
            "masks": mask_out,
            "contours": contours,
            "contour_regressions": contour_regressions
        }

if __name__ == "__main__":
    print("")

    net = ComposeNet()
    net.cuda()
    net.train()

    fake_img = torch.rand(16, 3, 256, 256)
    fake_img = fake_img.cuda()

    output = net(fake_img)
    # print(output)

