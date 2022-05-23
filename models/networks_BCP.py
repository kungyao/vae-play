from copy import deepcopy
from math import log2

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

class ContentEndoer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # self.convs = nn.Sequential(
        #     Conv2d(in_channels, 64, 3, stride=2, bn=None, activate="lrelu"), 
        #     Conv2d(64, 128, 3, stride=2, bn=None, activate="lrelu"), 
        #     Conv2d(128, 256, 3, stride=2, bn=None, activate="lrelu"), 
        #     Conv2d(256, 512, 3, stride=2, bn=None, activate="lrelu"), 
        #     Conv2d(512, 1024, 3, stride=2, bn=None, activate="lrelu"), 
        #     Conv2d(1024, 2048, 3, stride=1, bn=None, activate="lrelu"), 
        #     Conv2d(2048, 2048, 3, stride=1, bn=None, activate="lrelu"), 
        # )
        self.conv_first = Conv2d(in_channels, 64, 3, stride=2, bn=None, activate="lrelu")

        # res34 = resnet34(pretrained=True)
        res34 = resnet34()
        self.backbone_0 = nn.Sequential(
            deepcopy(res34.layer1), 
            # initialize_model(Conv2d(64, 64, 3, stride=1, bn=None, activate="lrelu")), 
            deepcopy(res34.layer2), 
            # initialize_model(Conv2d(128, 128, 3, stride=1, bn=None, activate="lrelu")), 
            deepcopy(res34.layer3), 
            # initialize_model(Conv2d(256, 256, 3, stride=1, bn=None, activate="lrelu")), 
            deepcopy(res34.layer4), 
        )
        self.backbone_1 = nn.Sequential(
            # deepcopy(res34.layer1), 
            Conv2d(64, 128, 3, stride=2, bn=None, activate="lrelu"), 
            # deepcopy(res34.layer2), 
            Conv2d(128, 256, 3, stride=2, bn=None, activate="lrelu"), 
            # deepcopy(res34.layer3), 
            Conv2d(256, 512, 3, stride=2, bn=None, activate="lrelu"), 
            # deepcopy(res34.layer4), 
            Conv2d(512, 512, 3, stride=1, bn=None, activate="lrelu"), 
        )

        self.conv_last = Conv2d(512 * 2, 512 * 2, 1, stride=1, bn=None, activate=None)
        self.out_size = 16
        self.out_channels = 512 * 2

    def forward(self, x: torch.Tensor):
        x = self.conv_first(x)
        x_0 = self.backbone_0(x)
        x_1 = self.backbone_1(x)
        b, c, h, w = x_0.shape
        # weight_solid = cls_x[:, 0].reshape(b, 1, 1, 1).repeat(1, c, h, w) * weight_solid
        # weight_emit = cls_x[:, 1].reshape(b, 1, 1, 1).repeat(1, c, h, w) * weight_emit
        x = torch.cat([x_0, x_1], dim=1)
        x = self.conv_last(x)
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
        
        # self.frequency_encode = []
        # level = int(log2(image_size)) - 1
        # tmp_channel = in_channels
        # tmp_out_channel = in_channels
        # for _ in range(level):
        #     tmp_channel = tmp_out_channel
        #     tmp_out_channel = min(pt_size, tmp_channel * 2)
        #     self.frequency_encode.append(Conv2d(tmp_channel, tmp_out_channel, 3, stride=2, bn=None, activate="lrelu"))
        # self.frequency_encode.append(Conv2d(tmp_channel, pt_size, 1, stride=1, bn=None, activate=None))
        # self.frequency_encode.append(nn.AdaptiveAvgPool2d(1))
        # self.frequency_encode = nn.Sequential(*self.frequency_encode)
        # embed_size = 2 + 1
        # self.value_encoder = ValueEncoder(embed_size, in_channels, pt_size=pt_size)

        self.batch_attention = nn.Sequential(
            SelfAttentionBlock(pt_size), 
            SelfAttentionBlock(pt_size), 
            SelfAttentionBlock(pt_size)
        )
        # -10~10
        # -10~10
        # offset_x, offset_y
        in_channels = in_channels * (1) # Plus embedding
        self.params_pred = nn.Sequential(
            Linear(in_channels, in_channels, activate='lrelu'), 
            Linear(in_channels, in_channels, activate=None), 
            Linear(in_channels, 2, activate=None)
        )
        self.frequency_pred = nn.Sequential(
            Linear(in_channels, in_channels, activate='relu'), 
            Linear(in_channels, in_channels, activate='relu'), 
            Linear(in_channels, 1, activate=None),
            nn.Sigmoid()
        )

    def process(self, x: torch.Tensor, contours: torch.Tensor):
        dtype = x.dtype
        device = x.device
        b, c, h, w = x.shape
        resamples = []
        for i, cnt in enumerate(contours):
            normalized_pts = cnt.to(device)
            if normalized_pts.numel() != 0:
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
        resamples = torch.stack(resamples, dim=0)
        return resamples
    
    def forward(self, x: torch.Tensor, contours: torch.Tensor, cls_x:torch.Tensor):
        b, c, h, w = x.shape
        # Sample points on feature map. (batch, pt, in_channel(256)).
        x_pt_feature = self.process(x, contours) 
        # # (batch, c(=pt_size), 1, 1)
        # x_freq = self.frequency_encode(x)
        # cls_x = cls_x.reshape(b, 1, 1, -1).repeat(1, self.max_point, 1, 1)
        # x_freq_embed = torch.cat([x_freq, cls_x], dim=3)
        # x_freq_embed = self.value_encoder(x_freq_embed)

        # Do predict.
        x_pt_feature = x_pt_feature.reshape(b, self.max_point, c, 1)
        # print(x_pt_feature.shape, x_freq.shape, cls_x.shape)
        # x = torch.cat([x_pt_feature, x_freq_embed], dim=-1)
        x = x_pt_feature
        # print(x.shape)
        x = self.batch_attention(x)
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
        x_pred = self.params_pred(x) 
        x_freq = self.frequency_pred(x).squeeze()
        # print(x.shape)
        # x_freq = x_freq.squeeze()
        # tmp_freq = []
        # for i in range(b):
        #     tmp_freq.append(x_freq[i][:size_per_img[i]])
        # x_freq = torch.cat(tmp_freq, dim=0)
        return x_pred, x_freq

class ComposeNet(nn.Module):
    def __init__(self, image_size, pt_size=4096):
        super(ComposeNet, self).__init__()
        self.max_point = pt_size
        # 實心or射線
        self.cls_classifier = nn.Sequential(
            resnet18(pretrained=True), 
            nn.Linear(1000, 2)
        )

        self.add_coord = AddCoords(if_normalize=True)
        self.encoder = ContentEndoer(3 + 2)
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
            b, c, h, w = x.shape
            size = []
            contours = []
            for i in range(b):
                cnt = find_contour(b[i])
                cnt = resample_points(cnt, self.max_point)
                cnt = (cnt / h - 0.5) / 0.5
                size.append(len(cnt))
                contours.append(cnt)
            # contours = torch.stack(contours, dim=0)
        # 
        cls_x = self.cls_classifier(x)
        # 
        x = self.add_coord(x)
        x = self.encoder(x)
        x, x_freq = self.line_predictor(x, contours, cls_x.detach())
        x = x.split(size)
        x_freq = x_freq.split(size)
        return {
            "classes": cls_x, 
            "contours": contours, 
            "target_pts": x, 
            "target_frequency": x_freq, 
        }
