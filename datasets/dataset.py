import os
import sys
import random

import cv2
import json
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF 
from rdp import rdp
from PIL import Image, ImageChops
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from tools.utils import encode_circle_param, generate_circle_param, generate_circle_img
from tools.utils import find_contour, resample_points

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
    def __init__(self, img_size, if_rnadom_gen) -> None:
        super().__init__()
        self.img_size = img_size
        self.if_rnadom_gen = if_rnadom_gen
        if if_rnadom_gen:
            # Fill with white
            self.rd_rotation_fw = transforms.RandomRotation(30, fill=1.0)
            # Fill with black
            self.rd_rotation_fb = transforms.RandomRotation(30, fill=0.0)
            self.rd_vertical = transforms.RandomVerticalFlip()
            self.rd_horizontal = transforms.RandomHorizontalFlip()

    def do_operation(self, img, seed, is_white_bg):
        if img is not None:
            img = TF.to_tensor(img)
            img = TF.resize(img, self.img_size, interpolation=TF.InterpolationMode.NEAREST)
            if self.if_rnadom_gen:
                random.seed(seed)
                torch.manual_seed(seed)
                if is_white_bg:
                    img = self.rd_rotation_fw(img)
                else:
                    img = self.rd_rotation_fb(img)
                img = self.rd_vertical(img)
                img = self.rd_horizontal(img)
        return img

    def __call__(self, img, bimg, eimg, contour, key_contour):
        # Set random seed for doing same operation for input data.
        seed = random.randint(0, 2147483647)
        # Do
        img = self.do_operation(img, seed, False)
        bimg = self.do_operation(bimg, seed, False)
        eimg = self.do_operation(eimg, seed, False)

        contour = torch.FloatTensor(contour)
        key_contour = torch.FloatTensor(key_contour)

        return img, bimg, eimg, contour, key_contour

# Bubble & Edge
class BEDataset(Dataset):
    def __init__(self, data_path, img_size, if_test=False) -> None:
        super().__init__()
        self.imgs = []
        self.masks = []
        self.labels = []
        self.transform = BTransform((img_size[1], img_size[0]), True)

        self.if_test = if_test
        for cls_name in os.listdir(data_path):
            if not if_test:
                if cls_name not in ["1", "2", "3"]:
                    continue
            else:
                if cls_name not in ["test"]:
                    continue
            
            cls_folder = os.path.join(data_path, cls_name)
            for patch in os.listdir(cls_folder):
                if "layer" in patch or "mask" in patch or "edge" in patch or "bubble" in patch:
                    continue
                name, ext = patch.split(".")[:2]
                self.imgs.append(os.path.join(cls_folder, f"{name}.{ext}"))

                if not if_test:
                    self.labels.append(int(cls_name))
                    self.masks.append(os.path.join(cls_folder, f"{name}_layer.{ext}"))
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # 
        img = Image.open(self.imgs[idx], "r").convert("RGB")
        
        if not self.if_test:
            # 
            mask = Image.open(self.masks[idx], "r").convert("RGB")
            mask = np.array(mask)
            bg = np.where((mask[:,:,0]==255) & (mask[:,:,1]==255) & (mask[:,:,2]==255))
            mask[bg] = (0, 0, 0)
            # 
            eimg = mask[:, :, 1]
            bimg = mask[:, :, 0]
            label = self.labels[idx]
        else:
            bimg = None
            eimg = None
            label = None
        #
        img, bimg, eimg, _, _ = self.transform(img, bimg, eimg, [], [])
        return img, bimg, eimg, label

# Bubble & Contour
class BCDataset(Dataset):
    def __init__(self, data_path, img_size, padding=1, max_points=256, ifTest=False, debug=None) -> None:
        self.imgs = []
        self.bimgs = []
        self.eimgs = []
        self.contours = []
        self.key_contours = []
        self.transform = BTransform((img_size[1], img_size[0]), False)
        self.ifTest = ifTest

        for cls_name in os.listdir(data_path):
            cls_folder = os.path.join(data_path, cls_name)
            for patch in os.listdir(cls_folder):
                # if "mask" in patch and "mask_edge" not in patch:
                #     self.imgs.append(os.path.join(cls_folder, patch))
                if "mask" in patch or "edge" in patch or "bubble" in patch:
                    continue
                item = patch.split(".")
                name = item[0]
                ext = item[1]
                # non-text and external noise
                # self.imgs.append(os.path.join(cls_folder, patch))
                # original image
                self.imgs.append(os.path.join(cls_folder, f"{name}_edge.{ext}"))
                self.bimgs.append(os.path.join(cls_folder, f"{name}_mask.{ext}"))
                self.eimgs.append(os.path.join(cls_folder, f"{name}_mask_edge.{ext}"))
                if debug is not None:
                    if len(self.imgs) >= debug:
                        break
            if debug is not None:
                if len(self.imgs) >= debug:
                    break
        
        if not self.ifTest:
            self.preprocess(img_size, padding, max_points)
        else:
            for i in range(len(self.imgs)):
                self.contours.append([])
                self.key_contours.append([])
    
    def preprocess(self, img_size, padding, max_points):
        for b_path in self.bimgs:
            bimg = Image.open(b_path, "r").convert("RGB")
            # 預先resize，計算reszie後的contour。
            bimg = bimg.resize(img_size, resample=Image.NEAREST)
            bimg = np.array(bimg)
            white = np.where((bimg[:,:,0]==255) & (bimg[:,:,1]==255) & (bimg[:,:,2]==255))
            bimg[white] = (0, 0, 0)
            bimg = bimg[:,:,0]
            bimg = np.pad(bimg, ((padding, padding), (padding, padding)), constant_values=(0, ))
            contour = find_contour(bimg)
            self.key_contours.append(rdp(contour, epsilon=4))
            self.contours.append(resample_points(contour, max_points=max_points))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx], "r").convert("RGB")
        bimg = Image.open(self.bimgs[idx], "r").convert("RGB")
        eimg = Image.open(self.eimgs[idx], "r").convert("RGB")
        # Process bubble mask
        bimg = np.array(bimg)
        white = np.where((bimg[:,:,0]==255) & (bimg[:,:,1]==255) & (bimg[:,:,2]==255))
        bimg[white] = (0, 0, 0)
        bimg = bimg[:,:,0]
        # Process edge mask
        eimg = np.array(eimg)
        white = np.where((eimg[:,:,0]==255) & (eimg[:,:,1]==255) & (eimg[:,:,2]==255))
        eimg[white] = (0, 0, 0)
        eimg = eimg[:,:,0]
        # Transform
        img, bimg, eimg, cnt, key_cnt = self.transform(img, bimg, eimg, self.contours[idx], self.key_contours[idx])
        return img, bimg, eimg, cnt, key_cnt

# 
class BEDatasetGAN(Dataset):
    def __init__(self, data_path, img_size, if_test=False) -> None:
        super().__init__()
        self.imgs = []
        self.masks = []
        self.labels = []
        self.transform = BTransform((img_size[1], img_size[0]), True)

        self.if_test = if_test
        for cls_name in os.listdir(data_path):
            if cls_name not in ["2", "3"]:
                 continue
            cls_label = int(cls_name)
            if cls_label == 1 or cls_label == 2:
                cls_label = 1
            elif cls_label == 3:
                cls_label = 2
            cls_folder = os.path.join(data_path, cls_name)
            for patch in os.listdir(cls_folder):
                if "layer" in patch or "mask" in patch or "edge" in patch or "bubble" in patch:
                    continue
                name, ext = patch.split(".")[:2]
                # self.imgs.append(os.path.join(cls_folder, f"{name}_bubble.{ext}"))
                self.imgs.append(os.path.join(cls_folder, f"{name}_mask2.{ext}"))

                self.labels.append(cls_label - 1)
                self.masks.append(os.path.join(cls_folder, f"{name}_layer.{ext}"))
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # 
        img = Image.open(self.imgs[idx], "r").convert("RGB")
        label = self.labels[idx]
        # 
        mask = Image.open(self.masks[idx], "r").convert("RGB")
        mask = np.array(mask)
        bg = np.where((mask[:,:,0]==255) & (mask[:,:,1]==255) & (mask[:,:,2]==255))
        mask[bg] = (0, 0, 0)
        # 
        # eimg = mask[:, :, 1]
        eimg = None
        bimg = mask[:, :, 0]
        #
        img, bimg, eimg, _, _ = self.transform(img, bimg, eimg, [], [])
        bimg = bimg.repeat(3, 1, 1)
        # tmp_eimg = eimg.repeat(3, 1, 1)
        # img = torch.multiply(img, tmp_eimg) + (1 - tmp_eimg)
        # img = torch.multiply(img, eimg) + (1 - eimg)
        return img, bimg, label

# Bubble parameter
class BPDataset(Dataset):
    def __init__(self, data_path, img_size) -> None:
        super().__init__()
        self.imgs = []
        self.layers = []
        self.ellipses = []
        self.infos = []
        self.img_size = img_size
        self.preprocess(data_path)

    def preprocess(self, data_path):
        img_path = os.path.join(data_path, "img")
        layer_path = os.path.join(data_path, "layer")
        ellipse_path = os.path.join(data_path, "ellipse")
        annotation_path = os.path.join(data_path, "annotation")
        for name in os.listdir(img_path):
            name = name.split(".")[0]

            self.imgs.append(os.path.join(img_path, f"{name}.png"))
            self.layers.append(os.path.join(layer_path, f"{name}.png"))
            self.ellipses.append(os.path.join(ellipse_path, f"{name}.png"))

            with open(os.path.join(annotation_path, f"{name}.txt"), 'r') as fp:
                annotation = json.load(fp)
            data = {
                "center_x": annotation["center_x"], 
                "center_y": annotation["center_y"], 
                "radius_x": annotation["radius_x"], 
                "radius_y": annotation["radius_y"], 
                "step": annotation["step"], 
                "image_size": annotation["image_size"], 
            }
            samples = []
            for sample in annotation["samples"]:
                # x, y, nx, ny, length
                samples.append(sample)
            data["samples"] = samples
            self.infos.append(data)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image_mode = "RGB" # "L"
        img = Image.open(self.imgs[idx], "r").convert(image_mode)
        # scale = self.img_size / img.height
        scale = 1 / img.height
        img = img.resize((self.img_size, self.img_size)) # , resample=Image.NEAREST

        mask = Image.open(self.layers[idx], "r").convert("RGB")
        mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST) # 
        mask = np.array(mask)
        bg = np.where((mask[:,:,0]==255) & (mask[:,:,1]==255) & (mask[:,:,2]==255))
        mask[bg] = (0, 0, 0)
        mask = mask[:, :, 0]

        ellipse = Image.open(self.ellipses[idx], "r").convert(image_mode)
        ellipse = ellipse.resize((self.img_size, self.img_size)) # , resample=Image.NEAREST
        data = self.infos[idx]
        target = {}
        phase1 = np.array([
            (data["center_x"] * scale - 0.5) / 0.5, 
            (data["center_y"] * scale - 0.5) / 0.5, 
            # data["center_x"] * scale, 
            # data["center_y"] * scale, 
            data["radius_x"] * scale, 
            data["radius_y"] * scale, 
            data["step"]
        ])
        phase2 = np.array(data["samples"])
        # phase2[:, 0] = phase2[:, 0]
        phase2[:, 1] = (phase2[:, 1] * scale - 0.5) / 0.5
        phase2[:, 2] = (phase2[:, 2] * scale - 0.5) / 0.5
        # phase2[:, 1] = phase2[:, 1] * scale
        # phase2[:, 2] = phase2[:, 2] * scale
        # phase2[:, 3] = phase2[:, 3]
        # phase2[:, 4] = phase2[:, 4]
        phase2[:, 5] = phase2[:, 5] * scale
        # phase2[:, 6] = phase2[:, 6]
        # 
        img = TF.to_tensor(img)
        bmask = TF.to_tensor(mask)
        ellipse = TF.to_tensor(ellipse)
        target["phase1"] = torch.FloatTensor(phase1)
        target["phase2"] = torch.FloatTensor(phase2)
        return img, bmask, ellipse, target
