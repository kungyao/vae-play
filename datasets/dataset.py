import os
from scipy.signal.signaltools import resample

import cv2
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF 
from PIL import Image
from torch.utils.data import Dataset

from tools.utils import encode_circle_param, generate_circle_param, generate_circle_img

CHANNEL_SIZE = 1

class CDataset(Dataset):
    def __init__(self, n: int, min_radius:int =10, data_size:int =4096, ifGen=False, ifWrite=False) -> None:
        self.n = n
        self.ifGen = ifGen
        self.ifWrite = ifWrite

        if ifGen:
            self.imgs = None
            self.params = []
            for _ in range(data_size):
                self.params.append(generate_circle_param(n, min_radius))
            self.data_size = data_size
        else:
            data_dir = "./datas"
            self.imgs = []
            self.params = []
            for f in os.listdir(data_dir):
                self.imgs.append(os.path.join(data_dir, f))
                _, r, x, y = f.split("_")
                y = y.split(".")[0]
                self.params.append({
                    "radius": int(r),
                    "x": int(x), 
                    "y": int(y), 
                })
            self.data_size = len(self.imgs)
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        param = self.params[idx]
        if self.ifGen:
            img = generate_circle_img(self.n, param["x"], param["y"], param["radius"], channel_size=CHANNEL_SIZE)
            if self.ifWrite:
                cv2.imwrite(f"./datas/{idx}_{int(param['radius'])}_{int(param['x'])}_{int(param['y'])}.png", img)
        else:
            img = Image.open(self.imgs[idx], "r")
            if CHANNEL_SIZE == 1:
                img = img.convert("L")
            elif CHANNEL_SIZE == 3:
                img = img.convert("RGB")
            h, _ = img.size
            if h > self.n:
                img = img.resize((self.n, self.n))
        img = TF.to_tensor(img)
        return img, param
    
    @staticmethod
    def train_collate_fn(batch):
        imgs, params = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        img_size = imgs.shape[-1]
        rs = []
        xs = []
        ys = []
        for param in params:
            rs.append(param['radius'])
            xs.append(param['x'])
            ys.append(param['y'])
        rs = torch.FloatTensor(rs)
        xs = torch.FloatTensor(xs)
        ys = torch.FloatTensor(ys)

        params = encode_circle_param(img_size, rs, xs, ys)
        rs = params["radius"]
        xs = params["x"]
        ys = params["y"]

        target = torch.stack([rs, xs, ys], dim=-1)
        return imgs, target


class BDataset(Dataset):
    def __init__(self, data_path, transform=None, num_classes=None, ifTest=False, debug=None) -> None:
        self.imgs = []
        self.bimgs = []
        self.transform = transform
        self.ifTest = ifTest

        for cls_name in os.listdir(data_path):
            if num_classes is not None:
                if int(cls_name) >= num_classes:
                    continue
            cls_folder = os.path.join(data_path, cls_name)
            for patch in os.listdir(cls_folder):
                # if "mask" in patch and "mask_edge" not in patch:
                #     self.imgs.append(os.path.join(cls_folder, patch))
                if "mask" in patch or "mask_edge" in patch or "ori" in patch:
                    continue
                item = patch.split(".")
                name = item[0]
                ext = item[1]
                # non-text and external noise
                # self.imgs.append(os.path.join(cls_folder, patch))
                # original image
                self.imgs.append(os.path.join(cls_folder, f"{name}_ori.{ext}"))
                self.bimgs.append(os.path.join(cls_folder, f"{name}_mask.{ext}"))
                if debug is not None:
                    if len(self.imgs) >= debug:
                        return
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx], "r").convert("RGB")
        bimg = Image.open(self.bimgs[idx], "r").convert("RGB")
        bimg = np.array(bimg)
        white = np.where((bimg[:,:,0]==255) & (bimg[:,:,1]==255) & (bimg[:,:,2]==255))
        bimg[white] = (0, 0, 0)
        bimg = bimg[:,:,0]
        # to 3 channel
        # img = np.stack([img, img, img], axis=-1)
        # expand_img = self.dilate(img, iteration=2)
        # img = expand_img - img
        if self.transform is not None:
            img = self.transform(img)
            bimg = self.transform(bimg)
        else:
            img = TF.to_tensor(img)
            bimg = TF.to_tensor(bimg)
        return img, bimg
 

