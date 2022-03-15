import torch
import torch.nn as nn
import numpy as np

try:
    from models.blocks import *
except:
    from blocks import *

IMAGE_CHANNEL = 3

class StyleEncoder(nn.Module):
    def __init__(self, z_dim, image_size, max_channels=1024):
        super(StyleEncoder, self).__init__()
        in_dim = IMAGE_CHANNEL
        out_dim = 64
        self.convs = [Conv2d(in_dim, out_dim, 5, 1, activate=None)]
        n_level = int(np.log2(image_size)) - 2
        for _ in range(n_level):       
            in_dim = out_dim
            out_dim = min(out_dim*2, max_channels)
            self.convs.append(Conv2d(in_dim, out_dim, 3, stride=2, bn="instance"))
        self.convs.append(Conv2d(out_dim, out_dim, 3, stride=2))
        self.convs.append(Conv2d(out_dim, out_dim, 3, stride=2))
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
            nn.ReLU()
        )
        
        self.cat_convs = nn.Sequential(
            Conv2d(out_channel*2, out_channel, 3),
            SCSEBlock(out_channel, reduction=4), 
            SCSEBlock(out_channel, reduction=4), 
            nn.ReLU() 
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

class myConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, bn=None, activate='relu'):
        super().__init__()
        self.conv_1 = Conv2d(in_channel, out_channel, kernel_size, stride, bn, activate)
        self.conv_2 = Conv2d(in_channel, out_channel, kernel_size, stride, bn, activate)
    
    def forward(self, x, label):
        return self.conv_1(x) * (1 - label) + self.conv_2(x) * label

class Generator(nn.Module):
    def __init__(self, image_size, z_dim, max_channels=256):
        super().__init__()
        self.z_dim = z_dim
        self.image_size = image_size

        # self.conv = nn.Sequential(
        #     Conv2d(IMAGE_CHANNEL + 1, 32, 3, 1, activate=None),
        #     Conv2d(32, 32, 3, 1, activate=None),
        # )

        self.conv1 = myConv2d(IMAGE_CHANNEL + 1, 32, 3, 1, activate=None)
        self.conv2 = myConv2d(32, 32, 3, 1, activate=None)
        
        self.down1 = myConv2d(32, 64, 4, 2, bn="instance")
        self.down2 = myConv2d(64, 128, 4, 2, bn="instance")
        self.down3 = myConv2d(128, 256, 4, 2, bn="instance")
        self.down4 = myConv2d(256, 256, 4, 2, bn="instance")

        # self.up1 = StyleUp(256+256, 256)
        self.up1 = StyleUp(256, 256)
        self.up2 = StyleUp(256, 128)
        self.up3 = StyleUp(128, 64)
        # self.up4 = StyleUp(64, 32)

        # self.skip1 = Conv2d(256+256, 256, 3, 1, bn="instance")
        # self.skip2 = Conv2d(128+128, 128, 3, 1, bn="instance")
        # self.skip3 = Conv2d(64+64, 64, 3, 1, bn="instance")
        # self.skip4 = Conv2d(32+32, 32, 3, 1, bn="instance")
        self.skip1 = Conv2d(256, 256, 3, 1, bn="instance")
        self.skip2 = Conv2d(128, 128, 3, 1, bn="instance")
        self.skip3 = Conv2d(64, 64, 3, 1, bn="instance")
        # self.skip4 = Conv2d(32, 32, 3, 1, bn="instance")

        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.InstanceNorm2d(32), 
            Conv2d(32, 32, 3, 1, bn=None), 
            Conv2d(32, 32, 3, 1, bn=None), 
            Conv2d(32, IMAGE_CHANNEL, 3, 1, bn=None, activate=None),
            nn.Tanh()
        )
        
        # self.mlp0 = MLP(z_dim, 256, 3)
        # self.mlp1 = MLP(z_dim, 256, 3)
        # self.mlp2 = MLP(z_dim, 128, 3)
        # self.mlp3 = MLP(z_dim, 64, 3)
        # self.mlp4 = MLP(z_dim, 32, 3)
        self.mlp = MLP(z_dim, image_size * image_size, 3)
    
    def encode(self, x, style_code, labels):
        style_code = self.mlp(style_code)
        style_code = style_code.reshape(style_code.size(0), 1, self.image_size, self.image_size)
        x = torch.cat([x, style_code], dim=1)
        
        labels = labels.reshape(labels.size(0), 1, 1, 1)
        x = self.conv2(self.conv1(x, labels), labels)
        d1 = self.down1(x, labels)
        d2 = self.down2(d1, labels)
        d3 = self.down3(d2, labels)
        d4 = self.down4(d3, labels)
        # d5 = self.down5(d4)
        return x, d1, d2, d3, d4 # , d5

    def decode(self, c0, d1, d2, d3, d4, style_code):
    # def decode(self, d1, d2, d3, d4, d5, style_code):
        # style_code0 = self.mlp0(style_code)
        # style0 = tile_like(style_code0, d4)
        # d5_ = torch.cat([d4, style0], dim=1)

        # style_code1 = self.mlp1(style_code)
        # style1 = tile_like(style_code1, d3)
        # skip1 = torch.cat([d3, style1], dim=1)
        skip1 = self.skip1(d3)
        up1 = self.up1(d4, skip1)

        # style_code2 = self.mlp2(style_code)
        # style2 = tile_like(style_code2, d2)
        # skip2 = torch.cat([d2, style2], dim=1)
        skip2 = self.skip2(d2)
        up2 = self.up2(up1, skip2)

        # style_code3 = self.mlp3(style_code)
        # style3 = tile_like(style_code3, d1)
        # skip3 = torch.cat([d1, style3], dim=1)
        skip3 = self.skip3(d1)
        up3 = self.up3(up2, skip3)

        # style_code4 = self.mlp4(style_code)
        # style4 = tile_like(style_code4, c0)
        # skip4 = torch.cat([c0, style4], dim=1)
        # skip4 = self.skip4(c0)
        # up4 = self.up4(up3, skip4)
        
        x = self.final(up3)
        return x

    def forward(self, x, style_code, labels):
        c0, d1, d2, d3, d4 = self.encode(x, style_code, labels)
        x = self.decode(c0, d1, d2, d3, d4, style_code)
        return x

class MLP(nn.Module):
    def __init__(self, nf_in, nf_out, num_blocks):
        super(MLP, self).__init__()
        self.model = []
        in_dim = nf_in
        out_dim = nf_in
        self.model.append(Linear(in_dim, out_dim, activate=None))
        ratio = int(2 ** (int(np.log2(nf_out / nf_in)) / (num_blocks - 1)))
        for _ in range(num_blocks - 2):
            in_dim = out_dim
            out_dim = min(in_dim*ratio, nf_out)
            self.model.append(Linear(in_dim, out_dim, activate=None))
        self.model.append(Linear(out_dim, nf_out, activate=None))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, image_size, num_of_classes, max_channels=256):
        super().__init__()
        in_dim = IMAGE_CHANNEL * 2
        out_dim = 64
        self.convs = [Conv2d(in_dim, out_dim, 5, 1)]
        n_level = int(np.log2(image_size)) - 2
        for _ in range(n_level):       
            in_dim = out_dim
            out_dim = min(out_dim*2, max_channels)
            self.convs.append(Conv2d(in_dim, out_dim, 3, stride=2, bn="instance"))
        self.convs = nn.Sequential(*self.convs)

        self.adv_convs = nn.Sequential(
            Conv2d(out_dim, out_dim, 3, stride=2, activate="lrelu"), 
            Conv2d(out_dim, 1, 3, stride=2, activate=None)
        )

        self.aux_convs = nn.Sequential(
            Conv2d(out_dim, out_dim, 3, stride=2, activate="lrelu"), 
            Conv2d(out_dim, num_of_classes, 3, stride=2, activate=None)
        )

    def forward(self, x, x_content, y):
        x = torch.cat([x, x_content], dim=1)
        x = self.convs(x)
        adv_res = self.adv_convs(x).reshape(x.size(0), -1).sigmoid()
        aux_res = self.aux_convs(x).reshape(x.size(0), -1).softmax(dim=-1)
        return adv_res, aux_res


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
    

