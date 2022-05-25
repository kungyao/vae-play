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

LABEL_EMBED = 256
STYLE_EMBED = 256

class EmbedingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_size):
        super().__init__()
        self.embeding = nn.Embedding(in_channels, out_channels)
        self.convs = nn.Sequential(
            Linear(out_channels, out_channels, activate="lrelu"), 
            Linear(out_channels, out_channels, activate="lrelu"), 
        )

    def forward(self, x):
        x = self.embeding(x)
        x = self.convs(x)
        return x

class StyleEncodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_size):
        super().__init__()
        min_channel = 64
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
        min_channel = 64
        max_channel = 512
        min_in_size = int(np.power(2, 2))
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

        self.style_encoder = ParameterEmbedingNet(StyleEncodeBlock, in_size, in_type="image")
        relay_in = in_channels * min_in_size * min_in_size
        self.relay_convs = nn.Sequential(
            Linear(relay_in + LABEL_EMBED + STYLE_EMBED, relay_in), 
            Linear(relay_in, relay_in), 
        )

        # self.latent_guiding_group = 4
        # 2, 1, 1
        # relay_in = relay_in // self.latent_guiding_group
        # self.label_classify = nn.Sequential(
        #     Linear(relay_in, relay_in // 2), 
        #     Linear(relay_in // 2, relay_in // 4), 
        #     Linear(relay_in // 4, 143, activate=None), 
        #     nn.Softmax()
        # )
        # self.style_classify = nn.Sequential(
        #     Linear(relay_in, relay_in // 2), 
        #     Linear(relay_in // 2, relay_in // 4), 
        #     Linear(relay_in // 4, 2, activate=None), 
        #     nn.Softmax()
        # )

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
        y_cls, y_cnt_style = self.style_encoder(x, x)
        
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
        # x_logits = x
        # x_logits = x_logits.reshape(x_logits.size(0), -1)
        # # group_size = x_logits.size(1) // self.latent_guiding_group
        # # x_label_cls = self.label_classify(x_logits[:, -group_size*2:-group_size])
        # # x_style_cls = self.style_classify(x_logits[:, -group_size:])
        # x_label_cls = self.label_classify(x_logits)
        # x_style_cls = self.style_classify(x_logits)
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
        output = {
            "edges": edge_out,
            "masks": mask_out, 
            # "latent_label_cls": x_label_cls, 
            # "latent_style_cls": x_style_cls, 
        }
        if self.training:
            output["y_cls"] = y_cls
            output["y_cnt_style"] = y_cnt_style
        return output

class Discriminator(nn.Module):
    def __init__(self, in_size, in_channels, num_of_classes):
        super().__init__()
        self.conv_first = Conv2d(in_channels, 64, 3, stride=2, bn="instance", activate="lrelu")
        self.backbone = nn.Sequential(
            Conv2d(64, 128, 3, stride=2, bn="instance", activate="lrelu"), 
            Conv2d(128, 256, 3, stride=2, bn="instance", activate="lrelu"), 
            Conv2d(256, 256, 3, stride=2, bn="instance", activate="lrelu"),
            Conv2d(256, 512, 3, stride=2, bn="instance", activate="lrelu"),
        )

        self.embeding_block = ParameterEmbedingNet(EmbedingBlock, in_size, in_type="embed")

        in_size = in_size // 32
        in_size = 512 * in_size * in_size

        self.adv_convs = nn.Sequential(
            Linear(in_size + LABEL_EMBED + STYLE_EMBED, in_size // 2, activate="lrelu"), 
            Linear(in_size // 2, in_size // 4, activate="lrelu"), 
            Linear(in_size // 4, 1, activate=None)
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
        return adv_res, y_cls, y_cnt_style

if __name__ == "__main__":
    print("")

    net = ComposeNet()
    net.cuda()
    net.train()

    fake_img = torch.rand(16, 3, 256, 256)
    fake_img = fake_img.cuda()

    output = net(fake_img)
    # print(output)

