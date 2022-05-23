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

LABEL_EMBED = 128
STYLE_EMBED = 128

class EmbedingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_size):
        super().__init__()
        self.embeding = nn.Embedding(in_channels, out_channels)
        self.convs = nn.Sequential(
            Linear(out_channels, out_channels, activate="lrelu")
        )

    def forward(self, x):
        x = self.embeding(x)
        x = self.convs(x)
        return x

class StyleEncodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_size):
        super().__init__()
        min_channel = 32
        max_channel = out_channels
        repeat_num = int(np.log2(in_size)) - 3

        self.convs = [Conv2d(in_channels, min_channel, 3, stride=2, bn="instance")]
        in_channels = min_channel
        out_channels = min(in_channels * 2, max_channel)
        for _ in range(repeat_num): 
            self.convs.append(Conv2d(in_channels, out_channels, 3, stride=2, bn="instance"))
            in_channels = out_channels
            out_channels = min(in_channels * 2, max_channel)
        self.convs.append(Conv2d(in_channels, max_channel, 1, stride=1, bn="instance"))
        self.convs.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.convs = nn.Sequential(*self.convs)

    def forward(self, x):
        x = self.convs(x)
        x = x.reshape(x.size(0), -1)
        return x

class ParameterEmbedingNet(nn.Module):
    def __init__(self, encode_block, in_size, in_type=None):
        super().__init__()

        if in_type == "image":
            self.label_encode_block = encode_block(3, LABEL_EMBED, in_size)
            self.style_encode_block = encode_block(3, STYLE_EMBED, in_size)
        elif in_type == "embed":
            self.label_encode_block = encode_block(143, LABEL_EMBED, in_size)
            self.style_encode_block = encode_block(2, STYLE_EMBED, in_size)

    def forward(self, y_cls, y_cnt_style):
        y_cls = self.label_encode_block(y_cls)
        y_cnt_style = self.style_encode_block(y_cnt_style)
        return y_cls, y_cnt_style

class MaskNet(nn.Module):
    def __init__(self, in_channel):
        super(MaskNet, self).__init__()
        # # 
        # self.conv1 = Up(in_channel, in_channel // 4, if_add_coord=True)
        # # 
        # self.conv2 = Up(in_channel // 4, in_channel // 8, if_add_coord=True)
        # 
        self.out_channels = 1

        # self.attention_convs = nn.Sequential(
        #     Conv2d(in_channel, in_channel, 3, stride=1, bn="instance"), 
        #     Conv2d(in_channel, in_channel, 1, stride=1, bn="instance"), 
        #     Conv2d(in_channel, in_channel, 3, stride=1, bn="instance")
        # )

        self.predictor = nn.Sequential(
            Conv2d(in_channel, in_channel, 3, stride=1, bn="instance"), 
            Conv2d(in_channel, in_channel, 3, stride=1, bn="instance"), 
            Conv2d(in_channel, self.out_channels, 3, stride=1, bn=None, activate=None)
        )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x_attn = self.attention_convs(x)
        # x = x * x_attn
        x = self.predictor(x)
        return x

class EdgeNet(MaskNet):
    def __init__(self, in_channel):
        super(EdgeNet, self).__init__(in_channel)

    def forward(self, x):
        x = super(EdgeNet, self).forward(x)
        return x

class ComposeNet(nn.Module):
    def __init__(self, in_size):
        super(ComposeNet, self).__init__()
        # Feature extract
        min_channel = 32
        max_channel = 256
        min_in_size = int(np.power(2, 3))
        repeat_num = int(np.log2(in_size // min_in_size))
        
        self.down = nn.ModuleList()
        self.down.append(Conv2d(3, min_channel, 3, stride=1, bn="instance"))
        in_channels = min_channel
        out_channels = min(in_channels * 2, max_channel)
        for _ in range(repeat_num):
            self.down.append(nn.Sequential(
                Conv2d(in_channels, out_channels, 3, stride=2, bn="instance"), 
                Conv2d(out_channels, out_channels, 3, stride=1, bn="instance")
            ))
            in_channels = out_channels
            out_channels = min(in_channels * 2, max_channel)

        self.embeding_block = ParameterEmbedingNet(EmbedingBlock, in_size, in_type="embed")
        self.style_encoder = ParameterEmbedingNet(StyleEncodeBlock, in_size, in_type="image")
        relay_in = in_channels * min_in_size * min_in_size
        self.relay_convs = nn.Sequential(
            Linear(relay_in + LABEL_EMBED + STYLE_EMBED, relay_in), 
            Linear(relay_in, relay_in), 
        )

        self.up = nn.ModuleList()
        self.skip = nn.ModuleList()
        self.cat = nn.ModuleList()
        in_channels = min_channel
        out_channels = min(in_channels * 2, max_channel)
        for _ in range(repeat_num):
            self.up.append(Up(out_channels, in_channels))
            # self.up.append(Conv2d(out_channels, in_channels, 3, stride=1, bn="instance"))
            self.skip.append(Conv2d(in_channels, in_channels, 3, stride=1, bn="instance"))
            self.cat.append(Conv2d(in_channels * 2, in_channels, 3, stride=1, bn="instance"))
            in_channels = out_channels
            out_channels = min(in_channels * 2, max_channel)

        # Generate content mask
        self.mask_net = MaskNet(min_channel)
        # Generate edge mask
        self.edge_net = EdgeNet(min_channel)

    def forward(self, x, y=None):
        # 
        if y is not None:
            mid_idx = x.size(0) // 2
            y_emb_cls, y_emb_cnt_style = self.embeding_block(y["cls"][:mid_idx], y["cnt_style"][:mid_idx])
            y_enc_cls, y_enc_cnt_style = self.style_encoder(x[mid_idx:], x[mid_idx:])
            y_cls = torch.cat([y_emb_cls, y_enc_cls], dim=0)
            y_cnt_style = torch.cat([y_emb_cnt_style, y_enc_cnt_style], dim=0)
        else:
            y_cls, y_cnt_style = self.style_encoder(x, x)
        # 
        down_feats = []
        for m in self.down:
            x = m(x)
            down_feats.append(x)
        # 
        b, c, h, w = x.shape
        x = x.reshape(b, -1)
        x = torch.cat([x, y_cls, y_cnt_style], dim=1)
        x = self.relay_convs(x)
        x = x.reshape(b, c, h, w)
        # 
        for i in range(len(self.up)):
            idx = len(self.up) - 1 - i
            x_up = self.up[idx](x)
            x_skip = self.skip[idx](down_feats[len(down_feats) - 2 - i])
            x_cat = torch.cat([x_up, x_skip], dim=1)
            x = self.cat[idx](x_cat)
        
        # Predict mask
        mask_out = self.mask_net(x)
        # Predict edge
        edge_out = self.edge_net(x)
        # # 
        # font_cls_out = self.font_cls(feature)
        # content_style_cls_out = self.content_style_cls(feature)
        return {
            "edges": edge_out,
            "masks": mask_out, 
            # "labels": font_cls_out, 
            # "content_style": content_style_cls_out
        }

class Discriminator(nn.Module):
    def __init__(self, in_size, in_channels, num_of_classes):
        super().__init__()
        self.conv_first = Conv2d(in_channels, 32, 3, stride=2, bn="instance", activate="lrelu")
        self.backbone = nn.Sequential(
            Conv2d(32, 64, 3, stride=2, bn="instance", activate="lrelu"), 
            Conv2d(64, 128, 3, stride=2, bn="instance", activate="lrelu"), 
            Conv2d(128, 256, 3, stride=2, bn="instance", activate="lrelu"),
        )

        self.embeding_block = ParameterEmbedingNet(EmbedingBlock, in_size, in_type="embed")

        in_size = in_size // 16
        in_size = 256 * in_size * in_size

        self.adv_convs = nn.Sequential(
            Linear(in_size + LABEL_EMBED + STYLE_EMBED, in_size, activate="lrelu"), 
            Linear(in_size, in_size, activate="lrelu"), 
            Linear(in_size, 1, activate=None)
        )

        # self.aux_convs = nn.Sequential(
        #     Linear(in_size * 2, in_size, activate="lrelu"), 
        #     Linear(in_size, in_size, activate="lrelu"), 
        #     Linear(in_size, 2, activate=None)
        # )

    def forward(self, x, y_cls, y_cnt_style):
        x = self.conv_first(x)
        x = self.backbone(x)
        x = x.reshape(x.size(0), -1)
        
        y_cls, y_cnt_style = self.embeding_block(y_cls, y_cnt_style)
        x = torch.cat([x, y_cls, y_cnt_style], dim=1)

        adv_res = self.adv_convs(x).sigmoid()
        # aux_res = self.aux_convs(x).softmax(dim=-1)
        return adv_res #, aux_res

if __name__ == "__main__":
    print("")

    net = ComposeNet()
    net.cuda()
    net.train()

    fake_img = torch.rand(16, 3, 256, 256)
    fake_img = fake_img.cuda()

    output = net(fake_img)
    # print(output)

