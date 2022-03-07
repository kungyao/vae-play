from modulefinder import Module
import torch
import torch.nn as nn
import numpy as np

from torchvision.models.resnet import resnet50

try:
    from models.blocks import *
except:
    from blocks import *

class StyleEncoder(nn.Module):
    def __init__(self, z_dim, image_size, max_channels=256):
        super(StyleEncoder, self).__init__()
        in_dim = 3
        out_dim = 64
        self.convs = [Conv2d(in_dim, out_dim, 7, 1)]
        n_level = int(np.log2(image_size))
        for _ in range(n_level):       
            in_dim = out_dim
            out_dim = min(out_dim*2, max_channels)
            self.convs.append(Conv2d(in_dim, out_dim, 3, stride=2))
        self.convs = nn.Sequential(*self.convs)
        
        fc_levels = 3
        self.fcs = [Linear(out_dim, z_dim, activate=None)]
        for _ in range(fc_levels-1):
            self.fcs.append(Linear(z_dim, z_dim, activate=None))
        self.fcs = nn.Sequential(*self.fcs)

    def forward(self, x):
        x = self.convs(x)
        x = x.reshape(x.size(0), -1)
        x = self.fcs(x)
        return x

class StyleUp(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.up_convs = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, 4, 2, 1),
            nn.InstanceNorm2d(out_channel), 
            nn.ReLU(),
            Conv2d(out_channel, out_channel, 3)
        )
        
        self.cat_convs = nn.Sequential(
            Conv2d(out_channel*2, out_channel, 3),
            Conv2d(out_channel, out_channel, 3),
            Conv2d(out_channel, out_channel, 3)
        )
    
    def forward(self, x, skip):
        x = self.up_convs(x)
        x = torch.cat([x, skip], dim=1)
        x = self.cat_convs(x)
        return x

def tile_like(x, img):
    x = x.reshape(x.size(0), x.size(1), 1, 1)
    x = x.repeat(1, 1, img.size(2), img.size(3))
    return x

class Generator(nn.Module):
    def __init__(self, z_dim, max_channels=256):
        super().__init__()
        self.conv = Conv2d(3, 32, 7, 1)
        
        self.down1 = Conv2d(32, 64, 4, 2)
        self.down2 = Conv2d(64, 128, 4, 2)
        self.down3 = Conv2d(128, 256, 4, 2)
        self.down4 = Conv2d(256, 256, 4, 2)
        self.down5 = Conv2d(256, 256, 4, 2)

        self.up1 = StyleUp(256+z_dim, 256)
        self.up2 = StyleUp(256, 256)
        self.up3 = StyleUp(256, 128)
        self.up4 = StyleUp(128, 64)

        self.skip1 = Conv2d(256+z_dim, 256, 3, 1)
        self.skip2 = Conv2d(256+z_dim, 256, 3, 1)
        self.skip3 = Conv2d(128+z_dim, 128, 3, 1)
        self.skip4 = Conv2d(64+z_dim, 64, 3, 1)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2d(64, 3, 7, 1),
            nn.Sigmoid()
        )
        
        self.mlp = MLP(z_dim, z_dim, 3)
    
    def encode(self, x):
        x = self.conv(x)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        return d1, d2, d3, d4, d5

    def decode(self, d1, d2, d3, d4, d5, style_code):
        style_code = self.mlp(style_code)
        
        style0 = tile_like(style_code, d5)
        d5_ = torch.cat([d5, style0], dim=1)

        style1 = tile_like(style_code, d4)
        skip1 = torch.cat([d4, style1], dim=1)
        skip1 = self.skip1(skip1)
        up1 = self.up1(d5_, skip1)

        style2 = tile_like(style_code, d3)
        skip2 = torch.cat([d3, style2], dim=1)
        skip2 = self.skip2(skip2)
        up2 = self.up2(up1, skip2)

        style3 = tile_like(style_code, d2)
        skip3 = torch.cat([d2, style3], dim=1)
        skip3 = self.skip3(skip3)
        up3 = self.up3(up2, skip3)

        style4 = tile_like(style_code, d1)
        skip4 = torch.cat([d1, style4], dim=1)
        skip4 = self.skip4(skip4)
        up4 = self.up4(up3, skip4)
        
        x = self.final(up4)
        return x

    def forward(self, x, style_code):
        d1, d2, d3, d4, d5 = self.encode(x)
        x = self.decode(d1, d2, d3, d4, d5, style_code)
        return x

class MLP(nn.Module):
    def __init__(self, nf_in, nf_out, num_blocks):
        super(MLP, self).__init__()
        self.model = []
        self.model.append(Linear(nf_in, 512))
        for _ in range(num_blocks - 2):
            self.model.append(Linear(512, 512))
        self.model.append(Linear(512, nf_out, activate=None))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, image_size, num_of_classes, max_channels=256):
        super().__init__()
        in_dim = 3
        out_dim = 64
        self.convs = [Conv2d(in_dim, out_dim, 7, 1)]
        n_level = int(np.log2(image_size)) - 2
        for _ in range(n_level):       
            in_dim = out_dim
            out_dim = min(out_dim*2, max_channels)
            self.convs.append(Conv2d(in_dim, out_dim, 3, stride=2))

        self.convs.append(Conv2d(out_dim, out_dim, 3, stride=2, activate="lrelu"))
        self.convs.append(Conv2d(out_dim, num_of_classes, 3, stride=2, activate=None))
        self.convs = nn.Sequential(*self.convs)

    def forward(self, x, y):
        x = self.convs(x)
        x = x.reshape(x.size(0), -1)                          # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        x = x[idx, y]                                         # (batch)
        return x


if __name__ == "__main__":
    print("Test models with specific input.")

    z_dim = 512
    image_size = 256
    batch_size = 2

    gen = Generator(z_dim)
    stl_encoder = StyleEncoder(z_dim, 256)
    disc = Discriminator(image_size, 3)

    gen.cuda()
    stl_encoder.cuda()
    disc.cuda()

    fake_img = torch.rand(batch_size, 3, image_size, image_size)
    fake_img = fake_img.cuda()

    fake_label = torch.randint(0, 3, (batch_size, ))
    fake_label = fake_label.cuda()

    # with torch.no_grad():
    style_code = stl_encoder(fake_img)
    gen_img = gen(fake_img, style_code)
    logit = disc(fake_img, fake_label)
    print(logit)
    
