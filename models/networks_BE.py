import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

try:
    from models.blocks import *
except:
    from blocks import *

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
        self.out_channels = 1
        self.predictor = nn.Sequential(
            Conv2d(in_channel, in_channel // 2, 3, stride=1, bn=False, activate=None), 
            Conv2d(in_channel // 2, self.out_channels, 3, stride=1, bn=False, activate=None)
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

class EdgeNet(MaskNet):
    def __init__(self, in_channel):
        super(EdgeNet, self).__init__(in_channel)

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

class ComposeNet(nn.Module):
    def __init__(self):
        super(ComposeNet, self).__init__()
        # Feature extract
        self.feature_net = FeatureNet()
        # Generate initial mask region
        self.mask_net = MaskNet(self.feature_net.out_channels+2)
        # Generate edge mask (?) according to mask net.
        self.edge_net = EdgeNet(self.feature_net.out_channels+2)
        # Expand two new channel for coordinate
        self.add_coords = AddCoords()

        # Initialize parameter
        self.initialize(self.mask_net)

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
        feature = self.feature_net(x)
        feature = self.add_coords(feature)
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

