import torch
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, bn=True, activate='relu'):
        super(Conv2d, self).__init__()
        bias = not bn
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
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if activate is not None:
            if activate == 'relu':
                conv.append(nn.ReLU())
            elif activate == 'lrelu':
                conv.append(nn.LeakyReLU(0.2))
            elif activate == 'tanh':
                conv.append(nn.Tanh())
        self.conv = nn.Sequential(*conv)

    def forward(self, input):
        return self.conv(input)

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
        self.q = Conv2d(in_channel, in_channel//8, 1, bn=False)
        self.k = Conv2d(in_channel, in_channel//8, 1, bn=False)
        self.v = Conv2d(in_channel, in_channel, 1, bn=False)
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

class ResidualBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResidualBlock, self).__init__()
        self.res_conv = Conv2d(in_channel, in_channel, kernel_size=1, bn=True, activate='lrelu')
        self.conv1 = Conv2d(in_channel, in_channel*2, kernel_size=1, bn=True, activate='lrelu')
        self.conv2 = Conv2d(in_channel*2, in_channel*2, kernel_size=3, bn=True, activate='lrelu')
        self.conv3 = Conv2d(in_channel*2, in_channel, kernel_size=1, bn=True, activate='lrelu')
        #parameter initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)
        out = self.res_conv(x) + residual
        return out

class AddCoords(nn.Module):
    def __init__(self):
        super(AddCoords, self).__init__()
    
    def forward(self, x):
        dtype = x.dtype
        device = x.device
        b, c, h, w = x.shape
        new_coord_i = torch.arange(0, w, dtype=dtype, device=device).reshape(1, 1, 1, -1).repeat(b, 1, h, 1)
        new_coord_j = torch.arange(0, h, dtype=dtype, device=device).reshape(1, 1, -1, 1).repeat(b, 1, 1, w)
        x = torch.cat([x, new_coord_i, new_coord_j], dim=1)
        return x

