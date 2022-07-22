from copy import deepcopy
from turtle import shape

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision.models.resnet import resnet18, resnet34
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
# from torchvision.models.segmentation import 

try:
    from models.blocks import *
except:
    from blocks import *

class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.backbone = resnet_fpn_backbone('resnet50', True)
        # 
        self.aux_convs = []
        target_out_channels = 32
        in_channels = self.backbone.out_channels
        repeat_num = int(np.log2(in_channels // target_out_channels))
        for _ in range(repeat_num):
            self.aux_convs.append(Conv2d(in_channels, in_channels// 2, 1, stride=1, bn="batch"))
            self.aux_convs.append(Conv2d(in_channels//2, in_channels//2, 3, stride=1, bn="batch"))
            in_channels = in_channels // 2
        self.aux_convs = nn.Sequential(*self.aux_convs)
        self.out_channels = target_out_channels

    def forward(self, x):
        # 0 : size / 4
        # 1 : size / 8
        # 2 : size / 16
        # 3 : size / 32
        # pool : size / 64
        x = self.backbone(x)["0"]
        x = self.aux_convs(x)
        return x

class MaskNet(nn.Module):
    def __init__(self, in_channel):
        super(MaskNet, self).__init__()
        # 
        self.conv1 = Up(in_channel, in_channel // 4, if_add_coord=True)
        # 
        self.conv2 = Up(in_channel // 4, in_channel // 8, if_add_coord=True)
        # 
        self.out_channels = 1
        self.predictor = nn.Sequential(
            Conv2d(in_channel // 8, in_channel // 4, 3, stride=1, bn=None, activate=None), 
            Conv2d(in_channel // 4, in_channel // 8, 3, stride=1, bn=None, activate=None), 
            Conv2d(in_channel // 8, self.out_channels, 3, stride=1, bn=None, activate=None)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.predictor(x)
        return x

class EdgeNet(MaskNet):
    def __init__(self, in_channel):
        super(EdgeNet, self).__init__(in_channel)

    def forward(self, x):
        x = super(EdgeNet, self).forward(x)
        return x

class ComposeNet(nn.Module):
    def __init__(self):
        super(ComposeNet, self).__init__()
        # Feature extract
        self.feature_net = FeatureNet()
        # Generate initial mask region
        self.mask_net = MaskNet(self.feature_net.out_channels)
        # Generate edge mask (?) according to mask net.
        self.edge_net = EdgeNet(self.feature_net.out_channels)
        # Expand two new channel for coordinate
        self.add_coords = AddCoords()

    def forward(self, x):
        feature = self.feature_net(x)
        # feature = self.add_coords(feature)
        # Predict mask
        mask_out = self.mask_net(feature)
        # Predict edge
        edge_out = self.edge_net(feature)
        return {
            "edges": edge_out,
            "masks": mask_out
        }

class MaskMapper(nn.Module):
    def __init__(self, in_channels, in_size, max_channel=128):
        super().__init__()
        min_in_size = int(np.power(2, 3))
        repeat_num = int(np.log2(in_size // min_in_size)) - 2

        self.convs = nn.Sequential(
            Conv2d(in_channels, 16, 3, 2, bn=None, activate='lrelu'), 
            Conv2d(16, 32, 3, 2, bn=None, activate='lrelu'), 
        )
        in_channels = 32
        out_channels = min(in_channels * 2, max_channel)
        self.feat_modules = nn.ModuleList()
        for _ in range(repeat_num):
            self.feat_modules .append(
                nn.Sequential(
                    Conv2d(in_channels, out_channels, 3, 2, bn='batch', activate='lrelu'), 
                    Conv2d(out_channels, out_channels, 3, 1, bn='batch', activate='lrelu'), 
                )
            )
            in_channels = out_channels
            out_channels = min(in_channels * 2, max_channel)
        self.pooler = nn.Sequential(
            Conv2d(in_channels, max_channel, 1, 1, bn=None, activate=None), 
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def forward(self, x: torch.Tensor, m: torch.Tensor):
        x = torch.cat([x, m], dim=1)
        # 
        x = self.convs(x)
        feat_list = []
        for idx, m in enumerate(self.feat_modules):
            x = m(x)
            feat_list.append(x.reshape(x.size(0), -1) * (idx // 2 + 1))
        # 
        feat_list = torch.cat(feat_list, dim=1)
        x = self.pooler(x)
        x = x.reshape(x.size(0), -1)
        return x, feat_list

class Discriminator(nn.Module):
    def __init__(self, in_channels, in_size, num_classes):
        super().__init__()
        max_channel = 64
        self.num_classes = num_classes
        self.content_disc = MaskMapper(2, in_size, max_channel=max_channel)
        self.boundary_disc = MaskMapper(2, in_size, max_channel=max_channel)
        
        self.predictor = nn.Sequential(
            Linear(max_channel*2, max_channel*2, bias=True, activate='lrelu'), 
            Linear(max_channel*2, max_channel, bias=True, activate='lrelu'), 
            Linear(max_channel, num_classes, bias=False, activate=None), 
        )

    def forward(self, x: torch.Tensor, m1: torch.Tensor, m2: torch.Tensor):
        x = x[:, 0, :, :].reshape(x.size(0), 1, x.size(2), x.size(3))
        # 
        x_m1, feats_m1 = self.content_disc(x, m1)
        # Boundary is black
        x_m2, feats_m2 = self.boundary_disc(x, m2)
        # 
        feats = torch.cat([feats_m1, feats_m2], dim=1)
        x = torch.cat([x_m1, x_m2], dim=1)
        x = self.predictor(x)
        return x, feats

if __name__ == "__main__":
    print("")

    net = ComposeNet()
    net.cuda()
    net.train()

    fake_img = torch.rand(16, 3, 256, 256)
    fake_img = fake_img.cuda()

    output = net(fake_img)
    # print(output)

