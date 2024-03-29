from operator import index
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
AUG_ROTATE = True

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
        # set background to white for training BE model.
        img = self.do_operation(img, seed, True)
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
                    self.masks.append(os.path.join(cls_folder, f"{name}_layer.{ext}"))
                    self.labels.append(int(cls_name))
    
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
            bimg = mask[:, :, 0]
            eimg = mask[:, :, 1]
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
    def __init__(self, data_path, img_size, if_test=False, select_list=None) -> None:
        super().__init__()
        self.imgs = []
        self.masks = []
        self.labels = []
        self.transform = BTransform((img_size[1], img_size[0]), True)

        self.if_test = if_test
        for cls_name in os.listdir(data_path):
            if select_list is not None:
                if cls_name not in select_list:
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
                # sx, sy, ex, ey, length, is_key
                samples.append(sample)
            data["samples"] = samples
            self.infos.append(data)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx], "r").convert("L")
        # scale = self.img_size / img.height
        scale = 1 / img.height
        img = img.resize((self.img_size, self.img_size)) # , resample=Image.NEAREST

        mask = Image.open(self.layers[idx], "r").convert("RGB")
        mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST) # 
        mask = np.array(mask)
        bg = np.where((mask[:,:,0]==255) & (mask[:,:,1]==255) & (mask[:,:,2]==255))
        mask[bg] = (0, 0, 0)
        bmask = mask[:, :, 0]
        emask = mask[:, :, 1]

        ellipse = Image.open(self.ellipses[idx], "r").convert("RGB")
        ellipse = ellipse.resize((self.img_size, self.img_size)) # , resample=Image.NEAREST
        data = self.infos[idx]
        target = {}
        phase1 = np.array([
            (data["center_x"] * scale - 0.5) / 0.5, 
            (data["center_y"] * scale - 0.5) / 0.5, 
            # data["center_x"] * scale, 
            # data["center_y"] * scale, 
            data["radius_x"] * scale / 0.5, 
            data["radius_y"] * scale / 0.5, 
            data["step"]
        ])
        phase2 = np.array(data["samples"])
        # phase2[:, 0] = phase2[:, 0]
        phase2[:, 1] = (phase2[:, 1] * scale - 0.5) / 0.5
        phase2[:, 2] = (phase2[:, 2] * scale - 0.5) / 0.5
        # phase2[:, 3] = phase2[:, 3]
        # phase2[:, 4] = phase2[:, 4]
        phase2[:, 5] = phase2[:, 5] * scale / 0.5
        # phase2[:, 6] = phase2[:, 6]
        # 
        img = TF.to_tensor(img)
        bmask = TF.to_tensor(bmask)
        emask = TF.to_tensor(emask)
        # 
        img = torch.cat([img, bmask, emask], dim=0)
        bmask = bmask.repeat(3, 1, 1)
        ellipse = TF.to_tensor(ellipse)
        target["phase1"] = torch.FloatTensor(phase1)
        target["phase2"] = torch.FloatTensor(phase2)
        return img, bmask, ellipse, target

class BPDatasetTEST(Dataset):
    def __init__(self, data_path, img_size) -> None:
        super().__init__()
        self.imgs = []
        self.masks = []
        self.transform = BTransform((img_size, img_size), False)

        for cls_name in os.listdir(data_path):
            if cls_name not in ["3"]:
                continue
            cls_folder = os.path.join(data_path, cls_name)
            for patch in os.listdir(cls_folder):
                if "layer" in patch or "mask" in patch or "edge" in patch or "bubble" in patch:
                    continue
                name, ext = patch.split(".")[:2]
                self.imgs.append(os.path.join(cls_folder, f"{name}_mask2.{ext}"))
                self.masks.append(os.path.join(cls_folder, f"{name}_layer.{ext}"))
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # 
        img = Image.open(self.imgs[idx], "r").convert("L")
        # 
        mask = Image.open(self.masks[idx], "r").convert("RGB")
        mask = np.array(mask)
        bg = np.where((mask[:,:,0]==255) & (mask[:,:,1]==255) & (mask[:,:,2]==255))
        mask[bg] = (0, 0, 0)
        # 
        bimg = mask[:, :, 0]
        eimg = mask[:, :, 1]
        #
        img, bimg, eimg, _, _ = self.transform(img, bimg, eimg, [], [])
        img = torch.cat([img, bimg, eimg], dim=0)
        bimg = bimg.repeat(3, 1, 1)
        # tmp_eimg = eimg.repeat(3, 1, 1)
        # img = torch.multiply(img, tmp_eimg) + (1 - tmp_eimg)
        # img = torch.multiply(img, eimg) + (1 - eimg)
        return img, bimg

def random_offset(bbox, img_size, maximum=None, offset=None):
    left, upper, right, lower = bbox
    # 
    left = left
    upper = upper
    right = img_size - right
    lower = img_size - lower
    # 
    if offset is not None:
        left = left + offset
        upper = upper + offset
        right = right + offset
        lower =   lower + offset
    #
    if maximum is not None:
        left = min(left, maximum)
        upper = min(upper, maximum)
        right = min(right, maximum)
        lower = min(lower, maximum)    
    # 
    left = -left + 1
    upper = -upper + 1
    # right = img_size - right
    # lower = img_size - lower
    offset_x = 0
    offset_y = 0
    if left < right:
        offset_x = np.random.randint(left, right)
    if upper < lower:
        offset_y = np.random.randint(upper, lower)
    return offset_x, offset_y

def resample_points_with_constraint(contour, max_points: int=256):
    l = len(contour)
    if l > max_points:
        # Select key first
        fix_select = contour[:, 5] >= 0.9
        random_select = np.where(~fix_select)[0]
        # Random select from non-key
        random_select_size = max_points - np.sum(fix_select)
        random_select_idx = np.arange(len(random_select))
        np.random.shuffle(random_select_idx)
        random_select_idx = random_select_idx[:random_select_size]
        # Merge two list
        fix_select[random_select[random_select_idx]] = True
        return np.array(contour[fix_select])
    return contour

# Bubble Contour parameter
class BCPDataset(Dataset):
    def __init__(self, data_path, img_size, max_points=256) -> None:
        self.layers = []
        self.masks = []
        self.labels = []
        self.annotations = []
        self.max_points = max_points

        for cls_name in os.listdir(data_path):
            cls_folder = os.path.join(data_path, cls_name)
            layer_path = os.path.join(cls_folder, "layers")
            mask_path = os.path.join(cls_folder, "masks")
            anno_path = os.path.join(cls_folder, "annotations")

            for name in os.listdir(layer_path):
                name = name.split(".")[0]

                self.labels.append(int(cls_name) - 1)
                self.layers.append(os.path.join(layer_path, f"{name}.png"))
                self.masks.append(os.path.join(mask_path, f"{name}.png"))

                with open(os.path.join(anno_path, f"{name}.txt"), 'r') as fp:
                    annotation = json.load(fp)
                annotation["points"] = np.array(annotation["points"], dtype=np.float32)
                self.annotations.append(annotation)

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, idx):
        mask = Image.open(self.masks[idx], "r").convert("L")
        layer = Image.open(self.layers[idx], "r").convert("RGB")
        # 
        width = mask.width
        height = mask.height
        random_rotation = 0.0
        if AUG_ROTATE:
            center_x = width * 0.5
            center_y = height * 0.5
            random_rotation = np.random.uniform(-15, 15)
            random_rotation_radian = random_rotation * np.pi / 180
        offset_x, offset_y = random_offset(mask.getbbox(), height)
        scale = 1 / height

        # 
        layer = np.array(layer)
        bg = np.where((layer[:,:,0]==255) & (layer[:,:,1]==255) & (layer[:,:,2]==255))
        layer[bg] = (0, 0, 0)
        bmask = layer[:, :, 0]
        emask = layer[:, :, 1]

        # 
        img = TF.to_tensor(mask)
        bmask = TF.to_tensor(bmask)
        emask = TF.to_tensor(emask)
        # 
        img = torch.cat([img, bmask, emask], dim=0)
        bmask = bmask.repeat(3, 1, 1)

        # sx, sy, ex, ey, l, key
        annotation = {}
        # points_annotation = torch.FloatTensor(resample_points_with_constraint(self.annotations[idx]["points"], max_points=self.max_points))
        points_annotation = self.annotations[idx]["points"].copy()

        if offset_x != 0 or offset_y != 0:
            img = TF.affine(img, angle=random_rotation, translate=[offset_x, offset_y], scale=1.0, shear=0.0, interpolation=Image.NEAREST)
            bmask = TF.affine(bmask, angle=random_rotation, translate=[offset_x, offset_y], scale=1.0, shear=0.0, interpolation=Image.NEAREST)
            # img = TF.affine(img, angle=0.0, translate=[offset_x, offset_y], scale=1.0, shear=0.0, interpolation=Image.NEAREST)
            # bmask = TF.affine(bmask, angle=0.0, translate=[offset_x, offset_y], scale=1.0, shear=0.0, interpolation=Image.NEAREST)

            if AUG_ROTATE:
                # rotate
                points_annotation[:, 0:3:2] = points_annotation[:, 0:3:2] - center_x
                points_annotation[:, 1:4:2] = points_annotation[:, 1:4:2] - center_y

                transform_x = points_annotation[:, 0:3:2] * np.cos(random_rotation_radian) - points_annotation[:, 1:4:2] * np.sin(random_rotation_radian)
                transform_y = points_annotation[:, 0:3:2] * np.sin(random_rotation_radian) + points_annotation[:, 1:4:2] * np.cos(random_rotation_radian)

                transform_x = transform_x + center_x
                transform_y = transform_y + center_y

                # transform_x[transform_x>=width] = width - 1
                # transform_x[transform_x<0] = 0

                # transform_y[transform_y>=height] = height - 1
                # transform_y[transform_y<0] = 0

                points_annotation[:, 0:3:2] = transform_x
                points_annotation[:, 1:4:2] = transform_y

            # offset
            if offset_x != 0:
                points_annotation[:, 0:3:2] += offset_x
            if offset_y != 0:
                points_annotation[:, 1:4:2] += offset_y


        points_annotation[:, :4] = (points_annotation[:, :4] * scale - 0.5) / 0.5
        # normalize length
        # points_annotation[:, 4] = (points_annotation[:, 4] * scale) / 0.5
        
        if torch.rand(1) < 0.5:
            img = TF.vflip(img)
            bmask = TF.vflip(bmask)
            points_annotation[:, 1:4:2] *= -1

        if torch.rand(1) < 0.5:
            img = TF.hflip(img)
            bmask = TF.hflip(bmask)
            points_annotation[:, 0:3:2] *= -1
        
        if AUG_ROTATE:
            select = np.logical_or(
                np.abs(points_annotation[:, 0]) <= 1, 
                np.logical_or(
                    np.abs(points_annotation[:, 1]) <= 1, 
                    np.logical_or(
                        np.abs(points_annotation[:, 2]) <= 1, 
                        np.abs(points_annotation[:, 3]) <= 1
                    )
                )
            )
            points_annotation = points_annotation[select]
        # Offset
        points_annotation[:, 2:4] = points_annotation[:, 2:4] - points_annotation[:, 0:2]

        annotation["points"] = torch.FloatTensor(resample_points_with_constraint(points_annotation, max_points=self.max_points))

        return img, bmask, self.labels[idx], annotation

class BCPDatasetTEST(Dataset):
    def __init__(self, data_path, img_size) -> None:
        super().__init__()
        self.imgs = []
        self.masks = []
        self.transform = BTransform((img_size, img_size), False)

        for cls_name in os.listdir(data_path):
            if cls_name not in ["2", "3"]:
                continue
            cls_folder = os.path.join(data_path, cls_name)
            for patch in os.listdir(cls_folder):
                if "layer" in patch or "mask" in patch or "edge" in patch or "bubble" in patch:
                    continue
                name, ext = patch.split(".")[:2]
                self.imgs.append(os.path.join(cls_folder, f"{name}_mask2.{ext}"))
                self.masks.append(os.path.join(cls_folder, f"{name}_layer.{ext}"))

        cls_folder = "D:/Manga/bubble-cnt-data/3"
        for cls_name in os.listdir(cls_folder):
            layer_path = os.path.join(cls_folder, "layers")
            mask_path = os.path.join(cls_folder, "masks")

            for name in os.listdir(layer_path):
                self.imgs.append(os.path.join(mask_path, f"{name}"))
                self.masks.append(os.path.join(layer_path, f"{name}"))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # 
        img = Image.open(self.imgs[idx], "r").convert("L")
        # 
        mask = Image.open(self.masks[idx], "r").convert("RGB")
        mask = np.array(mask)
        bg = np.where((mask[:,:,0]==255) & (mask[:,:,1]==255) & (mask[:,:,2]==255))
        mask[bg] = (0, 0, 0)
        bmask = mask[:, :, 0]
        emask = mask[:, :, 1]
        #
        img = TF.to_tensor(img)
        bmask = TF.to_tensor(bmask)
        emask = TF.to_tensor(emask)
        # 
        img = torch.cat([img, bmask, emask], dim=0)
        bmask = bmask.repeat(3, 1, 1)
        return img, bmask

def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    # y
    cmin, cmax = np.where(rows)[0][[0, -1]]
    # x
    rmin, rmax = np.where(cols)[0][[0, -1]]
    return rmin, cmin, rmax, cmax

class ImageDataset(Dataset):
    def __init__(self, manga_root_folder):
        self.imgs = []
        for manga in os.listdir(manga_root_folder):
            if manga not in ["AttackOnTitan", "DragonBall", "InitialD", "KurokosBasketball", "OnePiece"]:
                continue
            m_path = os.path.join(manga_root_folder, manga)
            for epi in os.listdir(m_path):
                m_e_path = os.path.join(m_path, epi)
                for chapter in os.listdir(m_e_path):
                    m_e_c_path = os.path.join(m_e_path, chapter)
                    # to manga page
                    origin_size_manga_folder = os.path.join(m_e_c_path, 'OriginSizeManga')
                    if not os.path.exists(origin_size_manga_folder):
                        continue
                    for page in os.listdir(origin_size_manga_folder):
                        self.imgs.append(os.path.join(origin_size_manga_folder, page))
    @staticmethod
    def collate_fn(batch):
        imgs = batch
        return imgs

    def __getitem__(self, index: int):
        img = Image.open(self.imgs[index]).convert('RGB')
        img = TF.to_tensor(img)
        return img

    def __len__(self):
        return len(self.imgs)

# Bubble & Edge
class BEGanDataset(Dataset):
    def __init__(self, data_path, img_size, if_test=False) -> None:
        super().__init__()
        self.imgs = []
        self.masks = []
        self.labels = []
        self.contours_content = []
        self.contours_boundary = []

        self.if_test = if_test
        self.img_size = img_size
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
                name = patch.split(".")[0]
                self.imgs.append(os.path.join(cls_folder, f"{name}.png"))

                if not if_test:
                    self.masks.append(os.path.join(cls_folder, f"{name}_layer.png"))
                    self.labels.append(int(cls_name))
                    with open(os.path.join(cls_folder, f"{name}.json"), 'r') as fp:
                        annotation = json.load(fp)
                    self.contours_content.append(np.array(annotation["points_content"], dtype=np.float32))
                    self.contours_boundary.append(np.array(annotation["points_boundary"], dtype=np.float32))
        self.synthesis_target = None

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # 
        img = Image.open(self.imgs[idx], "r").convert("RGB")
        width=  img.width
        height = img.height
        img = img.resize((self.img_size, self.img_size))
        img = TF.to_tensor(img)
        if not self.if_test:
            # 
            mask = Image.open(self.masks[idx], "r").convert("RGB")
            mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)
            mask = np.array(mask)
            bg = np.where((mask[:,:,0]==255) & (mask[:,:,1]==255) & (mask[:,:,2]==255))
            mask[bg] = (0, 0, 0)
            # 
            bimg = mask[:, :, 0]
            eimg = mask[:, :, 1]
            label = self.labels[idx]
            # bounding_box = bbox2(eimg)
            bounding_box = bbox2(bimg)
            # 
            bimg = TF.to_tensor(bimg)
            eimg = TF.to_tensor(eimg)
            contour_content = self.contours_content[idx].copy()
            contour_boundary = self.contours_boundary[idx].copy()

            # 
            center_x = width * 0.5
            center_y = height * 0.5
            random_scaling = np.random.uniform(1.0, 1.3) # random_rotation = 0.0
            random_rotation = np.random.uniform(-15, 15) # random_rotation = 0.0
            random_rotation_radian = random_rotation * np.pi / 180
            offset_x, offset_y = random_offset(bounding_box, self.img_size, maximum=50, offset=None)

            if offset_x != 0 or offset_y != 0:
                img = TF.affine(img, angle=random_rotation, translate=[offset_x, offset_y], scale=random_scaling, shear=0.0, interpolation=Image.NEAREST, fill=1.0)
                bimg = TF.affine(bimg, angle=random_rotation, translate=[offset_x, offset_y], scale=random_scaling, shear=0.0, interpolation=Image.NEAREST, fill=0.0)
                eimg = TF.affine(eimg, angle=random_rotation, translate=[offset_x, offset_y], scale=random_scaling, shear=0.0, interpolation=Image.NEAREST, fill=0.0)
                # 
                contour_content[:, 0] -= center_x
                contour_content[:, 1] -= center_y

                transform_x = contour_content[:, 0] * np.cos(random_rotation_radian) - contour_content[:, 1] * np.sin(random_rotation_radian)
                transform_y = contour_content[:, 0] * np.sin(random_rotation_radian) + contour_content[:, 1] * np.cos(random_rotation_radian)

                transform_x *= random_scaling
                transform_y *= random_scaling

                contour_content[:, 0] = transform_x + center_x + offset_x
                contour_content[:, 1] = transform_y + center_y + offset_y
                # 
                contour_boundary[:, 0] -= center_x
                contour_boundary[:, 1] -= center_y

                transform_x = contour_boundary[:, 0] * np.cos(random_rotation_radian) - contour_boundary[:, 1] * np.sin(random_rotation_radian)
                transform_y = contour_boundary[:, 0] * np.sin(random_rotation_radian) + contour_boundary[:, 1] * np.cos(random_rotation_radian)

                transform_x *= random_scaling
                transform_y *= random_scaling

                contour_boundary[:, 0] = transform_x + center_x + offset_x
                contour_boundary[:, 1] = transform_y + center_y + offset_y

            # to -1~-1
            contour_content[:, 0:2] = (contour_content[:, 0:2] / width - 0.5) / 0.5
            contour_boundary[:, 0:2] = (contour_boundary[:, 0:2] / width - 0.5) / 0.5

            if torch.rand(1) < 0.5:
                img = TF.vflip(img)
                bimg = TF.vflip(bimg)
                eimg = TF.vflip(eimg)
                # 
                contour_content[:, 1] *= -1
                contour_boundary[:, 1] *= -1

            if torch.rand(1) < 0.5:
                img = TF.hflip(img)
                bimg = TF.hflip(bimg)
                eimg = TF.hflip(eimg)
                # 
                contour_content[:, 0] *= -1
                contour_boundary[:, 0] *= -1

            select = np.logical_and(
                np.abs(contour_content[:, 0]) <= 1, 
                np.abs(contour_content[:, 1]) <= 1
            )
            contour_content = torch.FloatTensor(contour_content[select])
            select = np.logical_and(
                np.abs(contour_boundary[:, 0]) <= 1, 
                np.abs(contour_boundary[:, 1]) <= 1
            )
            contour_boundary = torch.FloatTensor(contour_boundary[select])
            
            if self.synthesis_target is not None:
                half = self.img_size // 2
                h, w = self.synthesis_target.shape[-2:]
                xmin = np.random.randint(half, w - half- 1) - half
                ymin = np.random.randint(half, h - half - 1) - half
                tmp_img = self.synthesis_target[:, ymin:ymin+self.img_size, xmin:xmin+self.img_size].clone().detach()
                total_mask = torch.logical_or(bimg, eimg).repeat(3, 1, 1)
                tmp_img[total_mask] = img[total_mask]
                img = tmp_img
                img = TF.gaussian_blur(img, 5)
        else:
            bimg = None
            eimg = None
            label = None
            contour_content = None
            contour_boundary = None
        return img, bimg, eimg, label, contour_content, contour_boundary
