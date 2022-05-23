import os
import math

import json
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF 
from PIL import Image, ImageChops
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageOps, ImageChops, ImageFilter

OPPOSITE_THRES = 0.5
MAX_ALLOWED_IOU = 0.1
MAX_ATTEMPTS_TO_SYNTHESIZE = 20

# Base image dataset.
class ImageDataset(Dataset):
    def __init__(self, image_list, debug=False):
        self.imgs = []
        self.targets = []

        with open(image_list, 'r') as f:
            data_sets = json.load(f)
        self.preprocessing(data_sets, debug)

    def preprocessing(self, data_sets, debug):
        for data in data_sets:
            page_folder = data['manga_folder']
            anno_path = data['annotation_path']
            # print(anno_path)
            with open(anno_path, 'r', encoding='utf-8') as f:
                annotation = json.load(f)
            width = annotation['imageWidth']
            height = annotation['imageHeight']
            occupied_boxes = []
            for shape in annotation['shapes']:
                label_name = shape['label']
                if label_name == 'Bubble' or label_name == 'Onomatopoeia-Kana':
                    pts = shape['points']
                    occupied_boxes.append([
                        max(min(pts[0][0], pts[1][0]), 0),      # xmin
                        max(min(pts[0][1], pts[1][1]), 0),      # ymin
                        min(max(pts[0][0], pts[1][0]), width),  # xmax
                        min(max(pts[0][1], pts[1][1]), height), # ymax
                    ])
            
            if len(occupied_boxes) != 0:
                self.imgs.append(os.path.join(page_folder, annotation['imagePath']))                
                area = width * height
                if data['data_type'] == 'manga109':
                    area /= 2
                target = {
                    'occupied_boxes': np.array(occupied_boxes),
                    "real_page_area": area
                }
                self.targets.append(target)

                if len(self.imgs) > 4 and debug:
                    break
            else:
                print(annotation['imagePath'])

    @staticmethod
    def collate_fn(batch):
        imgs, targets = zip(*batch)
        return imgs, targets

    def __getitem__(self, index: int):
        img = Image.open(self.imgs[index]).convert('L')
        img = img.point(lambda p: p>128 and 255)
        img = img.convert("RGB")
        target = self.targets[index]
        return img, target

    def __len__(self):
        return len(self.imgs)

def prepare_syhthesis_data(base_img: Image.Image, target, kana_imgs, kana_masks, augmentor):
    iw, ih = base_img.size
    page_area = target['real_page_area']
    occupied_boxes = target['occupied_boxes']

    train_imgs = []
    train_masks = []
    train_edge_masks = []
    train_content_styles = []
    for kana_img, kana_mask in zip(kana_imgs, kana_masks):
        kernel_size = int(round(np.random.uniform(8, 21), 0)) // 2
        params = {
            'scale' : np.random.uniform(0.707, 1.414),
            'angle' : np.random.uniform(-15, 15),
            'shear' : np.random.uniform(-0.3, 0.3),
            # round to digit zero
            'kernel_size' : kernel_size + (kernel_size+1)%2, 
            'p' : np.random.uniform(0.0, 1.0)
        }
        aug_img, aug_mask, aug_content_mask, aug_edge_mask = augmentor(kana_img, kana_mask, page_area, params)   

        aw, ah = aug_img.size
        center_x = aw // 2
        center_y = ah // 2

        xmin = np.random.randint(center_x, iw - center_x- 1, MAX_ATTEMPTS_TO_SYNTHESIZE) - center_x
        ymin = np.random.randint(center_y, ih - center_y - 1, MAX_ATTEMPTS_TO_SYNTHESIZE) - center_y

        tmp_boxes = np.stack([xmin, ymin, xmin + aw, ymin + ah], axis=1)
        area_new = (tmp_boxes[:, 2] - tmp_boxes[:, 0]) * (tmp_boxes[:, 3] - tmp_boxes[:, 1])

        area_ocp = (occupied_boxes[:, 2] - occupied_boxes[:, 0]) * (occupied_boxes[:, 3] - occupied_boxes[:, 1])

        lt = np.maximum(tmp_boxes[:, None, :2], occupied_boxes[:, :2])  # [N,M,2]
        rb = np.minimum(tmp_boxes[:, None, 2:], occupied_boxes[:, 2:])  # [N,M,2]

        wh = np.clip(rb - lt, 0, None)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area_new[:, None] + area_ocp - inter
        iou = inter / union
        iou_check = np.sum(iou <= MAX_ALLOWED_IOU, axis=1)

        # Make sure size is equal to kana_imgs.
        if np.sum(iou_check) == 0:
            box = tmp_boxes[0]
        else:      
            idx = np.argmax(iou_check)
            box = tmp_boxes[idx]
        
        train_img = base_img.crop(box)
        train_img.paste(aug_img, mask=aug_mask)
        train_imgs.append(train_img)
        train_masks.append(aug_content_mask)
        train_edge_masks.append(aug_edge_mask)
        train_content_styles.append(1 if params['p'] > 0.5 else 0)
        
    return train_imgs, train_masks, train_edge_masks, train_content_styles

class KanaImageDataset(Dataset):
    def __init__(self, image_folder):
        self.imgs = []
        for fp in os.listdir(image_folder):
            self.imgs.append(os.path.join(image_folder, fp))

    @staticmethod
    def collate_fn(batch):
        return batch
    
    def __getitem__(self, idx: int):
        img = Image.open(self.imgs[idx]).convert('L')
        img = img.point(lambda p: p>128 and 255)
        img = img.convert('RGB')
        img = ImageOps.expand(img, border=11, fill=(255, 255, 255))
        return img

    def __len__(self):
        return len(self.imgs)

# Make sure each do_x function result will trimmed on original image without any space.
class AugmentOperator(object):
    def __init__(self):
        self.initial_ratio = 0.018

    def adjust_initial_image_size(self, img, mask, target_area):
        w, h = img.size
        area = w * h
        scale = math.sqrt(self.initial_ratio * target_area / area)
        new_img, new_mask = self.do_scale(img, mask, scale)
        return new_img, new_mask

    @staticmethod
    def do_scale(img, mask, scale):
        w, h = img.size
        new_size = (int(w*scale), int(h*scale))
        new_img = img.resize(new_size, resample=Image.NEAREST)
        new_mask = mask.resize(new_size, resample=Image.NEAREST)
        return new_img, new_mask

    @staticmethod
    def do_rotate(img, mask, angle):
        new_img = img.rotate(angle, resample=Image.NEAREST, expand=True, fillcolor=(255, 255, 255))
        new_mask = mask.rotate(angle, resample=Image.NEAREST, expand=True)
        # new_img, new_mask = recalculate_bounding_box(new_img, new_mask)
        return new_img, new_mask

    @staticmethod
    def do_shear(img, mask, shear):
        def pil_space_filled_shear_(img_, shear, mode, fill):
            def calculate_size_(width, height, shear):
                # add abs for minus shear
                new_w = width + abs(int(shear*height))
                new_h = height
                return new_w, new_h
            w, h = img_.size
            new_w, new_h = calculate_size_(w, h, shear)
            new_img = Image.new(mode, (new_w, new_h), color=fill)
            # add if-else for minus shear
            new_img.paste(img_, ((new_w - w) if shear >= 0 else 0, 0))
            new_img = new_img.transform((new_w, new_h), Image.AFFINE, data=(1, shear, 0, 0, 1, 0), resample=Image.NEAREST, fillcolor=fill)
            return new_img
        new_img = pil_space_filled_shear_(img, shear, "RGB", (255, 255, 255))
        new_mask = pil_space_filled_shear_(mask, shear, "L", (0))
        # new_img, new_mask = recalculate_bounding_box(new_img, new_mask)
        return new_img, new_mask

    @staticmethod
    def do_white_edge(img, mask, kernel_size):
        # kernel_size should bigger than zero and odd.
        if kernel_size <= 0 or kernel_size%2 == 0:
            return img, mask
        # Padding image (kernel_size) pixel
        new_img = ImageOps.expand(img, border=kernel_size, fill=(255, 255, 255))
        new_mask = ImageOps.expand(mask, border=kernel_size)
        # Expand mask black pixel which mean the truly label area, so we can get white edge under original image.
        new_mask = new_mask.filter(ImageFilter.MaxFilter(kernel_size))
        return new_img, new_mask

    @staticmethod
    def recalculate_bounding_box(img, mask):
        # get non-zero bounding box.
        true_box = mask.getbbox()
        new_img = img.crop(true_box)
        new_mask = mask.crop(true_box)
        return new_img, new_mask

    @staticmethod
    def to_n_by_n(img, mask):
        w, h = img.size
        if w > h:
            anchor = (0, (w - h)//2)
            new_img = Image.new("RGB", (w, w), color=(255, 255, 255))
            new_img.paste(img, anchor)
            new_mask = Image.new("L", (w, w), color=(0))
            new_mask.paste(mask, anchor)
        elif h > w:
            anchor = ((h - w)//2, 0)
            new_img = Image.new("RGB", (h, h), color=(255, 255, 255))
            new_img.paste(img, anchor)
            new_mask = Image.new("L", (h, h), color=(0))
            new_mask.paste(mask, anchor)
        else:
            new_img = img
            new_mask = mask
        return new_img, new_mask

    @staticmethod
    def do_opposite(img, mask):
        tmp_mask = mask.convert("RGB")
        new_img = Image.new("RGB", img.size, color=(255, 255, 255))
        new_img = ImageChops.multiply(new_img, ImageChops.invert(tmp_mask))
        new_img = ImageChops.add(new_img, ImageChops.invert(img))
        return new_img, mask

    def __call__(self, img, mask, target_area, params):
        '''
        Parameters
        ----------
        img (PIL.Image): Character image.
        mask (PIL.Image): Character mask image.
        params (Dict[str]) : Augmentation parameters dictionary.
            {
                "scale" : (float)
                    Do scale coefficient.
                "angle" : (float)
                    Do rotate coefficient.
                "shear" : (float)
                    Do shear coefficient.
                "kernel_size" : (odd int)
                    Do white edge coefficient.
            }
        '''
        if 'scale' in params:
            img, mask = self.do_scale(img, mask, params['scale'])
        if 'angle' in params:
            img, mask = self.do_rotate(img, mask, params['angle'])
        if 'shear' in params:
            img, mask = self.do_shear(img, mask, params['shear'])
        # if 'kernel_size' in params:
        img, mask = self.do_white_edge(img, mask, params['kernel_size'])
        content_mask = ImageChops.invert(img.convert("L"))
        edge_mask = ImageChops.subtract(mask, content_mask)
        if 'p' in params:
            if params['p'] > OPPOSITE_THRES:
                img, mask = self.do_opposite(img, mask)

                img = ImageOps.expand(img, border=params['kernel_size'], fill=(255, 255, 255))
                mask = ImageOps.expand(mask, border=params['kernel_size'])
                mask = mask.filter(ImageFilter.MaxFilter(params['kernel_size']))
                content_mask = ImageOps.expand(content_mask, border=params['kernel_size'])
                edge_mask = ImageOps.expand(edge_mask, border=params['kernel_size'])

        # img, mask = self.adjust_initial_image_size(img, mask, target_area)
        w, h = img.size
        area = w * h
        scale = math.sqrt(self.initial_ratio * target_area / area)
        new_size = (int(w*scale), int(h*scale))
        img = img.resize(new_size, resample=Image.NEAREST)
        mask = mask.resize(new_size, resample=Image.NEAREST)
        content_mask = content_mask.resize(new_size, resample=Image.NEAREST)
        edge_mask = edge_mask.resize(new_size, resample=Image.NEAREST)

        # img, mask = self.recalculate_bounding_box(img, mask)
        true_box = mask.getbbox()
        img = img.crop(true_box)
        mask = mask.crop(true_box)
        content_mask = content_mask.crop(true_box)
        edge_mask = edge_mask.crop(true_box)

        def to_n_n(img: Image.Image, fill):
            w, h = img.size
            if w != h:
                if w > h:
                    anchor = (0, (w - h)//2)
                    new_size = w
                elif h > w:
                    anchor = ((h - w)//2, 0)
                    new_size = h
                new_img = Image.new(img.mode, (new_size, new_size), color=fill)
                new_img.paste(img, anchor)
                return new_img
            else:
                return img

        img = to_n_n(img, (255, 255, 255))
        mask = to_n_n(mask, (0))
        content_mask = to_n_n(content_mask, (0))
        edge_mask = to_n_n(edge_mask, (0))
        
        return img, mask, content_mask, edge_mask

PAGE_AREA = 8000 * 5000

# Font & Edge
class FEDataset(Dataset):
    # Only custom scan manga datas haave mask label. 
    # We implement this class on custom folder only.
    # @param manga_root_folder 
    # @param anno_root_folder 
    def __init__(self, debug=False):
        self.imgs = []
        self.labels = []
        self.preprocessing_fonts(debug)

    def preprocessing_fonts(self, debug):
        fonts_path = "./save_folder"
        for style in os.listdir(fonts_path):
            style_path = os.path.join(fonts_path, style)
            for c in os.listdir(style_path):
                label = int(c.split(".")[0]) + 1
                self.imgs.append(os.path.join(style_path, c))
                self.labels.append(label)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert('L')
        img = img.point(lambda p: p>128 and 255)
        mask = ImageChops.invert(img)
        img = img.convert("RGB")
        label = self.labels[idx]
        return img, mask, label
    
    @staticmethod
    def collate_fn(batch):
        imgs, masks, labels = zip(*batch)
        return imgs, masks, labels