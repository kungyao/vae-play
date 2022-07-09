
import torch
import torch.nn as nn
import numpy as np

try:
    from models.blocks import *
except:
    from blocks import *

class PartialEncoder(nn.Module):
    def __init__(self, in_channels, in_size, latent_length):
        super().__init__()
        max_channel = latent_length

        # 
        self.convs = []
        repeat_num = int(np.log2(in_size)) - 2
        self.convs = [Conv2d(in_channels, 32, 3, stride=1, bn="instance")]
        in_channels = 32
        out_channels = min(in_channels * 2, max_channel)
        for _ in range(repeat_num): 
            self.convs.append(Conv2d(in_channels, out_channels, 3, stride=2, bn="instance"))
            in_channels = out_channels
            out_channels = min(in_channels * 2, max_channel)
        self.convs.append(Conv2d(in_channels, max_channel, 1, stride=1, bn="instance"))
        self.convs.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.convs = nn.Sequential(*self.convs)

        # 
        self.linears = nn.Sequential(
            Linear(max_channel, latent_length, bias=True, activate='relu'), 
            Linear(latent_length, latent_length, bias=True, activate='relu'), 
            Linear(latent_length, latent_length, bias=False, activate=None)
        )
    
    def forward(self, x: torch.Tensor):
        # 
        x = self.convs(x)
        x = x.reshape(x.size(0), -1)
        x = self.linears(x)
        return x

class LatentMapper(nn.Module):
    def __init__(self, latent_size, target_size):
        super().__init__()
        latent_length = latent_size * latent_size
        if_upsample = target_size > latent_size
        if_donw_sample = target_size < latent_size
        repeat_num = int(np.log2(max(latent_size, target_size) // min(latent_size, target_size)))
        self.convs = []
        for _ in range(repeat_num):
            if if_upsample:
                self.convs.append(nn.Upsample(scale_factor=2, mode='bilinear'))
            self.convs.append(Conv2d(1, 1, 3, 2 if if_donw_sample else 1, bn="instance"))
        for _ in range(3):
            self.convs.append(Conv2d(1, 1, 3, 1, bn="instance"))

        self.linears = nn.Sequential(Linear(latent_length, latent_length), Linear(latent_length, latent_length))
        self.convs = nn.Sequential(*self.convs)
        self.latent_size = latent_size
    
    def forward(self, x):
        x = self.linears(x)
        x = x.reshape(x.size(0), 1, self.latent_size, self.latent_size)
        x = self.convs(x)
        return x

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
    def __init__(self, in_channels, in_size, latent_size):
        super().__init__()
        # Feature extract
        min_channel = 16
        max_channel = 32
        min_in_size = int(np.power(2, 4))
        repeat_num = int(np.log2(in_size // min_in_size))
        
        # 
        embed_type_count = 2
        latent_length = latent_size * latent_size
        up_size_list = []
        # 

        self.down = nn.ModuleList()
        self.down.append(Conv2d(3, min_channel, 3, stride=1, bn="batch"))
        cur_size = in_size
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
            cur_size = cur_size // 2
            up_size_list.append(cur_size)

        # 
        self.up = nn.ModuleList()
        self.skip = nn.ModuleList()
        self.cat = nn.ModuleList()
        in_channels = min_channel
        out_channels = min(in_channels * 2, max_channel)
        for _ in range(repeat_num):
            self.up.append(Up(out_channels + embed_type_count, in_channels))
            self.skip.append(Conv2d(in_channels, in_channels, 3, stride=1, bn="instance"))
            self.cat.append(Conv2d(in_channels * 2, in_channels, 3, stride=1, bn="instance"))
            in_channels = out_channels
            out_channels = min(in_channels * 2, max_channel)

        # 
        self.content_encoder = PartialEncoder(3, in_size, latent_length)
        self.boundary_encoder = PartialEncoder(3, in_size, latent_length)

        self.content_code_size_mapping = nn.ModuleList()
        self.boundary_code_size_mapping = nn.ModuleList()
        for up_size in up_size_list:
            self.content_code_size_mapping.append(LatentMapper(latent_size, up_size))
            self.boundary_code_size_mapping.append(LatentMapper(latent_size, up_size))
        
        # Generate content mask
        self.mask_net = MaskNet(min_channel)
        # Generate edge mask
        self.edge_net = EdgeNet(min_channel)

    def forward_latent(self, x: torch.Tensor, y=None):
        if y is None:
            content_code = self.content_encoder(x)
            boundary_code = self.boundary_encoder(x)
        else:
            # 
            c_mask = y[:, 0, :, :][:, None].repeat(1, 3, 1, 1)
            b_mask = y[:, 1, :, :][:, None].repeat(1, 3, 1, 1)
            # 
            content_code = self.content_encoder(c_mask)
            boundary_code = self.boundary_encoder(1.0 - b_mask)
        return content_code, boundary_code
    
    def forward_content_latent(self, x: torch.Tensor):
        content_code = self.content_encoder(x)
        return content_code

    def forward_boundary_latent(self, x: torch.Tensor):
        boundary_code = self.boundary_encoder(x)
        return boundary_code

    def forward(self, x: torch.Tensor, y=None):
        # 
        content_code, boundary_code = self.forward_latent(x, y=y)
        # 
        down_feats = []
        for m in self.down:
            x = m(x)
            down_feats.append(x)
        # 
        for i in range(len(self.up)):
            idx = len(self.up) - 1 - i
            # Cat with latent code
            level_content_code = self.content_code_size_mapping[idx](content_code)
            level_boundary_code = self.boundary_code_size_mapping[idx](boundary_code)
            
            x = torch.cat([x, level_content_code, level_boundary_code], dim=1)
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

class Discriminator(nn.Module):
    def __init__(self, in_channels, in_size):
        super().__init__()
        max_channel = 64
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

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # 
        x = x[:, 0, :, :].reshape(x.size(0), 1, x.size(2), x.size(3))
        x = torch.cat([x, y], dim=1)
        # 
        x = self.convs(x)
        # 
        feat_list = []
        for idx, m in enumerate(self.feat_modules):
            x = m(x)
            feat_list.append(x.reshape(x.size(0), -1) * (idx // 2 + 1))
        x = torch.cat(feat_list, dim=1)
        return x
