import os
from scipy.signal.signaltools import resample

import cv2
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF 
from rdp import rdp
from PIL import Image
from torch.utils.data import Dataset

from tools.utils import encode_circle_param, generate_circle_param, generate_circle_img
from tools.utils import find_contour

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

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((args.img_size, args.img_size), interpolation=TF.InterpolationMode.NEAREST),
#     # https://github.com/pytorch/vision/issues/9#issuecomment-304224800
#     # Should make sure that img and target do the same operation.
#     # 
#     # transforms.RandomRotation(10, fill=0.0), 
#     # transforms.RandomHorizontalFlip(),
#     # transforms.RandomVerticalFlip()
# ])
class BTransform(object):
    def __init__(self, img_size) -> None:
        super().__init__()
        self.img_size = img_size

    def __call__(self, img, bimg, contour, key_contour):
        img = TF.to_tensor(img)
        img = TF.resize(img, self.img_size, interpolation=TF.InterpolationMode.NEAREST)

        bimg = TF.to_tensor(bimg)
        bimg = TF.resize(bimg, self.img_size, interpolation=TF.InterpolationMode.NEAREST)
        
        contour = torch.FloatTensor(contour)
        key_contour = torch.FloatTensor(key_contour)

        return img, bimg, contour, key_contour

class BDataset(Dataset):
    def __init__(self, data_path, img_size, padding=1, ifTest=False, debug=None) -> None:
        self.imgs = []
        self.bimgs = []
        self.contours = []
        self.key_contours = []
        self.transform = BTransform((img_size[1], img_size[0]))
        self.ifTest = ifTest

        for cls_name in os.listdir(data_path):
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
                        break
            if debug is not None:
                if len(self.imgs) >= debug:
                    break
        
        if not self.ifTest:
            self.preprocess(img_size, padding)
        else:
            for i in range(len(self.imgs)):
                self.contours.append([])
                self.key_contours.append([])
    
    def preprocess(self, img_size, padding):
        for b_path in self.bimgs:
            bimg = Image.open(b_path, "r").convert("RGB")
            # 預先resize，計算reszie後的contour。
            bimg = bimg.resize(img_size, resample=Image.NEAREST)
            bimg = np.array(bimg)
            white = np.where((bimg[:,:,0]==255) & (bimg[:,:,1]==255) & (bimg[:,:,2]==255))
            bimg[white] = (0, 0, 0)
            bimg = bimg[:,:,0]
            bimg = np.pad(bimg, ((padding, padding), (padding, padding)), constant_values=(0, ))
            contour = find_contour(bimg, 256)
            self.contours.append(contour)
            self.key_contours.append(rdp(contour, epsilon=2))

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
        img, bimg, cnt, key_cnt = self.transform(img, bimg, self.contours[idx], self.key_contours[idx])
        return img, bimg, cnt, key_cnt
 

