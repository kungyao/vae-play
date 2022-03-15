import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

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

if __name__ == "__main__":
    print("")

    net = ComposeNet()
    net.cuda()
    net.train()

    fake_img = torch.rand(16, 3, 256, 256)
    fake_img = fake_img.cuda()

    output = net(fake_img)
    # print(output)

