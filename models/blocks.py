import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, bn=None, activate='relu'):
        super(Conv2d, self).__init__()
        bias = bn is None
        conv = [
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=bias,
            )
        ]
        if bn is not None:
            if bn == "batch":
                conv.append(nn.BatchNorm2d(out_channel, track_running_stats=False))
            elif bn == "instance":
                conv.append(nn.InstanceNorm2d(out_channel))
        if activate is not None:
            if activate == 'relu':
                conv.append(nn.ReLU())
            elif activate == 'lrelu':
                conv.append(nn.LeakyReLU(0.02))
            elif activate == 'tanh':
                conv.append(nn.Tanh())
        self.conv = nn.Sequential(*conv)

    def forward(self, input):
        return self.conv(input)

class Linear(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, activate='relu'):
        super(Linear, self).__init__()
        fc = [nn.Linear(in_channel, out_channel, bias=bias)]
        if activate is not None:
            if activate == 'relu':
                fc.append(nn.ReLU())
            elif activate == 'lrelu':
                fc.append(nn.LeakyReLU(0.2))
            elif activate == 'tanh':
                fc.append(nn.Tanh())
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        return self.fc(x)

class SCSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SCSEBlock, self).__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channel):
        super(SelfAttentionBlock, self).__init__()
        self.q = Conv2d(in_channel, in_channel//8, 1)
        self.k = Conv2d(in_channel, in_channel//8, 1)
        self.v = Conv2d(in_channel, in_channel, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        b, c, h, w = x.shape
        proj_query = self.q(x).view(b, -1, h*w).permute(0, 2, 1)
        proj_key = self.k(x).view(b, -1, h*w)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.v(x).view(b, -1, h*w)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        
        out = self.gamma*out + x
        return out #, attention

class AddCoords(nn.Module):
    def __init__(self, if_normalize=False):
        super(AddCoords, self).__init__()
        self.if_normalize = if_normalize
    
    def forward(self, x):
        dtype = x.dtype
        device = x.device
        b, c, h, w = x.shape
        new_coord_i = torch.arange(0, w, dtype=dtype, device=device).reshape(1, 1, 1, -1).repeat(b, 1, h, 1)
        new_coord_j = torch.arange(0, h, dtype=dtype, device=device).reshape(1, 1, -1, 1).repeat(b, 1, 1, w)
        if self.if_normalize:
            new_coord_i = (new_coord_i / w - 0.5) / 0.5
            new_coord_j = (new_coord_j / h - 0.5) / 0.5
        x = torch.cat([x, new_coord_i, new_coord_j], dim=1)
        return x

class Down(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, if_add_coord=False):
        super().__init__()
        self.if_add_coord = if_add_coord
        coord_channel = 2 if if_add_coord else 0
        self.conv = Conv2d(in_channel + coord_channel, out_channel, kernel_size, stride=2)
        if if_add_coord:
            self.add_coord = AddCoords()
    
    def forward(self, x):
        if self.if_add_coord:
            x = self.add_coord(x)
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channel, out_channel, if_add_coord=False):
        super().__init__()
        self.if_add_coord = if_add_coord
        coord_channel = 2 if if_add_coord else 0
        self.conv = nn.Sequential(
            Conv2d(in_channel + coord_channel, in_channel, 3, stride=1, bn="batch"), 
            Conv2d(in_channel, out_channel, 3, stride=1, bn="batch"), 
            Conv2d(out_channel, out_channel, 3, stride=1, bn="batch")
        )
        if if_add_coord:
            self.add_coord = AddCoords()

    def forward(self, x):
        if self.if_add_coord:
            x = self.add_coord(x)
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        return x

