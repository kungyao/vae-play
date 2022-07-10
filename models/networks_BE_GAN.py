
from tkinter.tix import Tree
import torch
import torch.nn as nn
import numpy as np

try:
    from models.blocks import *
except:
    from blocks import *

class MaskNet(nn.Module):
    def __init__(self, in_channel):
        super(MaskNet, self).__init__()
        self.out_channels = 1
        self.predictor = nn.Sequential(
            Conv2d(in_channel, in_channel, 3, stride=1, bn=None, activate='lrelu'), 
            Conv2d(in_channel, in_channel, 3, stride=1, bn=None, activate='lrelu'), 
            Conv2d(in_channel, self.out_channels, 3, stride=1, bn=None, activate=None)
        )

    def forward(self, x):
        x = self.predictor(x)
        return x

class EdgeNet(MaskNet):
    def __init__(self, in_channel):
        super(EdgeNet, self).__init__(in_channel)

    def forward(self, x):
        x = super(EdgeNet, self).forward(x)
        return x

class ComposeNet(nn.Module):
    def __init__(self, in_channels, in_size):
        super().__init__()
        # Feature extract
        min_channel = 32
        max_channel = 256
        min_in_size = int(np.power(2, 4))
        repeat_num = int(np.log2(in_size // min_in_size))
        
        # 
        self.down = nn.ModuleList()
        self.down.append(Conv2d(3, min_channel, 3, stride=1, bn="batch"))
        in_channels = min_channel
        out_channels = min(in_channels * 2, max_channel)
        for _ in range(repeat_num):
            self.down.append(nn.Sequential(
                Conv2d(in_channels, out_channels, 3, stride=2, bn="batch"), 
                Conv2d(out_channels, out_channels, 3, stride=1, bn="batch")
            ))
            in_channels = out_channels
            out_channels = min(in_channels * 2, max_channel)

        # 
        self.up = nn.ModuleList()
        self.skip = nn.ModuleList()
        self.cat = nn.ModuleList()
        in_channels = min_channel
        out_channels = min(in_channels * 2, max_channel)
        for _ in range(repeat_num):
            self.up.append(Up(out_channels, in_channels))
            self.skip.append(Conv2d(in_channels, in_channels, 3, stride=1, bn="instance"))
            self.cat.append(Conv2d(in_channels * 2, in_channels, 3, stride=1, bn="instance"))
            in_channels = out_channels
            out_channels = min(in_channels * 2, max_channel)

        # Generate content mask
        self.mask_net = MaskNet(min_channel)
        # Generate edge mask
        self.edge_net = EdgeNet(min_channel)

    def forward(self, x: torch.Tensor):
        # 
        down_feats = []
        for m in self.down:
            x = m(x)
            down_feats.append(x)
        # 
        for i in range(len(self.up)):
            idx = len(self.up) - 1 - i
            # Up-sampling
            x_up = self.up[idx](x)
            x_skip = self.skip[idx](down_feats[len(down_feats) - 2 - i])
            x_cat = torch.cat([x_up, x_skip], dim=1)
            x = self.cat[idx](x_cat)
            
        # Predict mask
        mask_out = self.mask_net(x)
        # Predict edge
        edge_out = self.edge_net(x)
        # 
        output = {
            "masks": mask_out, 
            "edges": edge_out
        }
        return output

class MaskMapper(nn.Module):
    def __init__(self, in_channels, in_size, max_channel=128):
        super().__init__()
        min_in_size = int(np.power(2, 4))
        repeat_num = int(np.log2(in_size // min_in_size))

        self.convs = nn.Sequential(
            Conv2d(in_channels, 16, 3, 2, bn=None, activate='lrelu'), 
            Conv2d(16, 32, 3, 2, bn=None, activate='lrelu'), 
        )
        in_channels = 32
        out_channels = min(in_channels * 2, max_channel)
        self.feat_modules = nn.ModuleList()
        for _ in range(repeat_num):
            self.feat_modules .append(Conv2d(in_channels, out_channels, 3, 2, bn='batch', activate='lrelu'))
            in_channels = out_channels
            out_channels = min(in_channels * 2, max_channel)
        self.pooler = nn.Sequential(
            Conv2d(in_channels, max_channel, 1, 1, bn=None, activate=None), 
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def forward(self, x: torch.Tensor, m: torch.Tensor):
        x = x[:, 0, :, :].reshape(x.size(0), 1, x.size(2), x.size(3))
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
        max_channel = 128
        self.num_classes = num_classes
        self.content_disc = MaskMapper(2, in_size, max_channel=max_channel)
        self.boundary_disc = MaskMapper(2, in_size, max_channel=max_channel)
    
        self.predictor = nn.Sequential(
            Linear(max_channel*2, max_channel*2, bias=True, activate='lrelu'), 
            Linear(max_channel*2, max_channel, bias=True, activate='lrelu'), 
            Linear(max_channel, num_classes, bias=False, activate=None), 
        )

    def forward(self, x: torch.Tensor, m1: torch.Tensor, m2: torch.Tensor):
        x_m1, feats_m1 = self.content_disc(x, m1)
        x_m2, feats_m2 = self.boundary_disc(x, m2)
        # 
        feats = torch.cat([feats_m1, feats_m2], dim=1)
        x = torch.cat([x_m1, x_m2], dim=1)
        x = self.predictor(x)
        return x, feats
