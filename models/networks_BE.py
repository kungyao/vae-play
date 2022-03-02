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
        out_channels = self.backbone.out_channels
        self.conv1 = Conv2d(out_channels, out_channels//2, 3, stride=1)
        self.conv2 = Conv2d(out_channels//2, out_channels//4, 3, stride=1)
        self.conv3 = Conv2d(out_channels//4, out_channels//8, 3, stride=1)
        self.out_channels = out_channels//8

    def forward(self, x):
        # 0 : size / 4
        # 1 : size / 8
        # 2 : size / 16
        # 3 : size / 32
        # pool : size / 64
        x = self.backbone(x)["0"]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
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
            Conv2d(in_channel // 8, in_channel // 4, 3, stride=1, bn=False, activate=None), 
            Conv2d(in_channel // 4, in_channel // 8, 3, stride=1, bn=False, activate=None), 
            Conv2d(in_channel // 8, self.out_channels, 3, stride=1, bn=False, activate=None)
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

class Discriminator(nn.Module):
    def __init__(self, types, img_size, max_conv_dim=256):
        super().__init__()
        self.backbone = []
        # Backbone output image size = 2**final_log_size
        final_log_size = 3
        repeat_num = int(np.log2(img_size)) - final_log_size - 1
        # Concate input mask and edge together, then get dim_in = 2.
        dim_in = 2
        dim_out = 32
        self.backbone.append(Conv2d(dim_in, dim_out, 5, stride=2))
        for _ in range(repeat_num):
            dim_in = dim_out
            dim_out = min(dim_in * 2, max_conv_dim)
            self.backbone.append(Conv2d(dim_in, dim_out, 5, stride=2))
        self.backbone = nn.Sequential(*self.backbone)

        self.adv_final = []
        self.aux_final = []
        for _ in range(final_log_size):
            self.adv_final.append(Conv2d(dim_out, dim_out, 3, stride=2))
            self.aux_final.append(Conv2d(dim_out, dim_out, 3, stride=2))
        self.adv_final += [Conv2d(dim_out, 1, 1, stride=1)]
        self.aux_final += [Conv2d(dim_out, types, 1, stride=1)]
        self.adv_final = nn.Sequential(*self.adv_final)
        self.aux_final = nn.Sequential(*self.aux_final)

    def forward(self, mask, edge):
        # Concate input
        x = torch.cat([mask, edge], dim=1)
        # 
        x = self.backbone(x)
        # 
        facticity = self.adv_final(x) # (batch, 1, )
        bubble_type = self.aux_final(x) # (batch, types, 1, 1)
        facticity = facticity.view(facticity.size(0), facticity.size(1)).sigmoid()
        bubble_type = bubble_type.view(bubble_type.size(0), bubble_type.size(1)).softmax(dim=-1)
        return facticity, bubble_type

if __name__ == "__main__":
    print("")

    net = ComposeNet()
    net.cuda()
    net.train()

    fake_img = torch.rand(16, 3, 256, 256)
    fake_img = fake_img.cuda()

    output = net(fake_img)
    # print(output)

