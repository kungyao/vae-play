from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn

from torch.nn.functional import grid_sample
from torchvision.models.resnet import resnet50
from torchvision.models.densenet import densenet121

try:
    from models.blocks import *
except:
    from blocks import *
from tools.utils import find_contour, resample_points

VALUE_WEIGHT = 512

class ContentEndoer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.convs = nn.Sequential(
            Conv2d(in_channels, 64, 3, stride=2, bn=None, activate="lrelu"), 
            Conv2d(64, 128, 3, stride=2, bn=None, activate="lrelu"), 
            Conv2d(128, 256, 3, stride=2, bn=None, activate="lrelu"), 
            Conv2d(256, 512, 3, stride=2, bn=None, activate="lrelu"), 
            Conv2d(512, 1024, 3, stride=2, bn=None, activate="lrelu"), 
            Conv2d(1024, 2048, 3, stride=1, bn=None, activate="lrelu"), 
            Conv2d(2048, 2048, 3, stride=1, bn=None, activate="lrelu"), 
        )
        self.out_channels = 2048

    def forward(self, x: torch.Tensor):
        x = self.convs(x)
        return x

class LinePredictor(nn.Module):
    def __init__(self, image_size, pt_size=4096, in_channels=256):
        super().__init__()
        self.max_point = pt_size
        self.batch_attention = nn.Sequential(
            SelfAttentionBlock(pt_size), 
            SelfAttentionBlock(pt_size), 
            SelfAttentionBlock(pt_size)
        )
        # -10~10
        # -10~10
        # offset_x, offset_y
        self.params_pred = nn.Sequential(
            Linear(in_channels, in_channels, activate='lrelu'), 
            Linear(in_channels, in_channels, activate=None), 
            Linear(in_channels, 2, activate=None)
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
    
    def forward(self, x: torch.Tensor, contours: torch.Tensor):
        # Sample points on feature map. (batch, pt, in_channel(256)).
        x = self.process(x, contours) 
        # Do predict.
        b, c, n = x.shape
        x = x.reshape(b, c, n, 1)
        x = self.batch_attention(x)
        x = x.reshape(b, c, n)
        size_per_img = [len(x) for x in contours]
        tmp_x = []
        for i in range(b):
            tmp_x.append(x[i][:size_per_img[i]])
        x = torch.cat(tmp_x, dim=0)
        # 
        x = self.params_pred(x) 
        return x

class ComposeNet(nn.Module):
    def __init__(self, image_size, pt_size=4096):
        super(ComposeNet, self).__init__()
        self.max_point = pt_size
        # 實心or射線
        self.cls_classifier = nn.Sequential(
            resnet50(pretrained=True), 
            nn.Linear(1000, 2)
        )

        self.add_coord = AddCoords(if_normalize=True)
        self.encoder = ContentEndoer(3 + 2)
        self.line_predictor = LinePredictor(image_size, pt_size=4096, in_channels=self.encoder.out_channels)

    def forward(self, x, target=None):
        if self.training and target is not None:
            size = []
            contours = []
            for t in target:
                size.append(len(t["total"]))
                contours.append(t["total"][:, :2])
            # contours = torch.stack(contours, dim=0)
        else:
            b = x.shape[0]
            size = []
            contours = []
            for i in range(b):
                cnt = find_contour(b[i])
                cnt = resample_points(cnt, self.max_point)
                size.append(len(cnt))
                contours.append(cnt)
            # contours = torch.stack(contours, dim=0)
        # 
        cls_x = self.cls_classifier(x)
        # 
        x = self.add_coord(x)
        x = self.encoder(x)
        x = self.line_predictor(x, contours)
        x = x.split(size)
        return {
            "classes": cls_x, 
            "contours": contours, 
            "target_pts": x, 
        }
