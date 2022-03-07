import os

import cv2
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from datasets .dataset import BCDataset, BEDataset, BEDatasetGAN
from tools.utils import makedirs


def viz_cnotours(contours, img_size, key_contours):
    imgs = []
    h, w = img_size
    for i in range(len(contours)):
        cnt = contours[i]
        img = torch.zeros(img_size)
        if cnt.numel() != 0:
            # cnt = torch.LongTensor(point_filter(cnt))
            cnt = cnt.to(dtype=torch.long)
            img[cnt[:, 1], cnt[:, 0]] = 1
            # for pt in cnt:
                # for step in steps:
                #     step_pt = pt + step
                #     if step_pt[0] >=0 and step_pt[0] < w and step_pt[1] >=0 and step_pt[1] < h:
                #         img[step_pt[1], step_pt[0]] = 1
        img = img.reshape(1, h, w).repeat(3, 1, 1)

        if key_contours is not None:
            kc = key_contours[i]
            kc = kc.to(dtype=torch.long)
            img[0, kc[:, 1], kc[:, 0]] = 1
            img[1, kc[:, 1], kc[:, 0]] = 0
            img[2, kc[:, 1], kc[:, 0]] = 0

        imgs.append(img)
    imgs = torch.stack(imgs, dim=0)
    return imgs

def BC_collate_fn(batch):
    imgs, bimgs, eimgs, cnt, key_cnt = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    bimgs = torch.stack(bimgs, dim=0)
    eimgs = torch.stack(eimgs, dim=0)
    return imgs, bimgs, eimgs, cnt, key_cnt

def test_BCDataset():
    padding = 1
    # width height
    dset = BCDataset("D:/code/python/manga-python-tools/results/bbb-single-train-type", (512, 512), padding=padding, debug=16)
    dloader = DataLoader(
        dset, 
        batch_size=8, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=BC_collate_fn, 
        pin_memory=True)
    
    res_output = "./tests"
    makedirs(res_output)

    for i, (imgs, bimgs, eimgs, contours, key_contours) in enumerate(dloader):
        b = imgs.size(0)
        h, w = imgs.shape[-2:]
        size = torch.Size([h+padding*2, w+padding*2])
        
        img_contours = viz_cnotours(contours, size, key_contours)
        imgs = F.pad(imgs.cpu(), (padding, padding, padding, padding), "constant", 0)
        bimgs = bimgs.repeat(1, 3, 1, 1)
        bimgs = F.pad(bimgs.cpu(), (padding, padding, padding, padding), "constant", 0)
        eimgs = eimgs.repeat(1, 3, 1, 1)
        eimgs = F.pad(eimgs.cpu(), (padding, padding, padding, padding), "constant", 0)

        vutils.save_image(
            torch.cat([imgs, bimgs, eimgs, img_contours], dim=0), 
            os.path.join(res_output, f"{i}.png"),
            nrow=b, 
            padding=2, 
            pad_value=1
        )

def BE_collate_fn(batch):
    imgs, bimgs, eimgs = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    bimgs = torch.stack(bimgs, dim=0)
    eimgs = torch.stack(eimgs, dim=0)
    return imgs, bimgs, eimgs

def test_BEDataset():
    # width height
    dset = BEDataset("D:/Manga/bubble-gen-label", (512, 512))
    dloader = DataLoader(
        dset, 
        batch_size=8, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=BE_collate_fn, 
        pin_memory=True)
    
    res_output = "./tests"
    makedirs(res_output)

    for i, (imgs, bimgs, eimgs) in enumerate(dloader):
        b = imgs.size(0)
        # h, w = imgs.shape[-2:]
        
        # imgs = F.pad(imgs.cpu(), (padding, padding, padding, padding), "constant", 0)
        bimgs = bimgs.repeat(1, 3, 1, 1)
        # bimgs = F.pad(bimgs.cpu(), (padding, padding, padding, padding), "constant", 0)
        eimgs = eimgs.repeat(1, 3, 1, 1)
        # eimgs = F.pad(eimgs.cpu(), (padding, padding, padding, padding), "constant", 0)

        vutils.save_image(
            torch.cat([imgs, bimgs, eimgs], dim=0), 
            os.path.join(res_output, f"{i}.png"),
            nrow=b, 
            padding=2, 
            pad_value=1
        )

def BEGAN_collate_fn(batch):
    imgs, bimgs, eimgs, labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    bimgs = torch.stack(bimgs, dim=0)
    eimgs = torch.stack(eimgs, dim=0)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    return imgs, bimgs, eimgs, labels

def tset_BEDatasetGAN():
    data_loader = DataLoader(
        BEDatasetGAN("D:/Manga/bubble-gen-label", (256, 256), if_test=True), 
        batch_size=16, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=BEGAN_collate_fn, 
        pin_memory=True)
    
    res_output = "./tests"
    makedirs(res_output)

    for i, (imgs, bimgs, eimgs, labels) in enumerate(data_loader):
        b = imgs.size(0)
        vutils.save_image(
            torch.cat([imgs], dim=0), 
            os.path.join(res_output, f"{i}.png"),
            nrow=b, 
            padding=2, 
            pad_value=1
        )

if __name__ == "__main__":
    # test_BCDataset()
    # test_BEDataset()
    tset_BEDatasetGAN()
    print()

