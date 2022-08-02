import os
import argparse

import cv2
import json
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
from scipy.ndimage.measurements import label as scipy_label

from models.networks_BE_font import ComposeNet
from tools.utils import makedirs

def paset_result_on_manga(img_path, ono_recon_info, ono_boxes, predictions, result_path, result_name, original_ono_boxes=None):
    pred_edges = predictions["edges"].cpu()
    pred_masks = predictions["masks"].cpu()
    b = pred_edges.size(0)

    pred_edges = pred_edges.cpu()
    pred_masks = pred_masks.cpu()

    pred_edges[pred_edges>=0.5] = 1
    pred_edges[pred_edges<0.5] = 0

    pred_masks[pred_masks>=0.5] = 1
    pred_masks[pred_masks<0.5] = 0

    img = TF.to_tensor(Image.open(img_path).convert("RGB")) * 255
    img = img.to(dtype=torch.uint8)
    c, h, w = img.shape

    # 0~255
    result_mask = torch.zeros(3, h, w, dtype=torch.uint8)
    # Check if occupied
    check_mask = torch.zeros(h, w, dtype=torch.bool)
    for i in range(b):
        # Create new mask for part saving
        merge_edge_mask = torch.zeros(h, w)
        merge_bubble_mask = torch.zeros(h, w)
        anchor_x, anchor_y, size = ono_recon_info[i].tolist()
        xmin, ymin, xmax, ymax = ono_boxes[i].tolist()
        width = xmax - xmin
        height = ymax - ymin
        
        tmp_edge = pred_edges[i]
        tmp_edge = F.interpolate(tmp_edge[None], size=(size, size), mode='nearest')[0]
        merge_edge_mask[ymin:ymax, xmin:xmax] = tmp_edge[0, anchor_y:anchor_y+height, anchor_x:anchor_x+width]
        # merge_edge_mask = merge_edge_mask.to(dtype=torch.bool)

        tmp_mask = pred_masks[i]
        tmp_mask = F.interpolate(tmp_mask[None], size=(size, size), mode='nearest')[0]
        merge_bubble_mask[ymin:ymax, xmin:xmax] = tmp_mask[0, anchor_y:anchor_y+height, anchor_x:anchor_x+width]
        # merge_bubble_mask = merge_bubble_mask.to(dtype=torch.bool)
        # 
        merge_edge_mask = merge_edge_mask.to(dtype=torch.bool)
        merge_bubble_mask = merge_bubble_mask.to(dtype=torch.bool)
        # Preserve Content Region and Remove Content Region From Edge Mask
        toatl_merge_mask = torch.logical_or(merge_edge_mask, merge_bubble_mask)
        toatl_merge_mask = toatl_merge_mask * ~check_mask
        check_mask = check_mask + toatl_merge_mask

        toatl_merge_mask = toatl_merge_mask.unsqueeze(dim=0).repeat(3, 1, 1)
        toatl_merge_mask = toatl_merge_mask.to(dtype=torch.uint8) * 255

        result_mask = result_mask + toatl_merge_mask
    result_mask[:, ~check_mask] = 255
    result_mask = result_mask.permute(1, 2, 0).numpy()
    cv2.imwrite(os.path.join(result_path, f"{result_name}.png"), result_mask)
    # # 
    # img[:, merge_edge_mask] = img[:, merge_edge_mask] // 2
    # img[0, merge_edge_mask] = img[0, merge_edge_mask] + 255 // 2
    # # 
    # img[:, merge_bubble_mask] = img[:, merge_bubble_mask] // 2
    # img[1, merge_bubble_mask] = img[1, merge_bubble_mask] + 255 // 2
    # img = img.to(dtype=torch.float) / 255
    # vutils.save_image(img, os.path.join("./results", f"{result_name}.png"))

    # result_mask = result_mask.to(dtype=torch.float)
    # vutils.save_image(result_mask, os.path.join(result_path, f"{result_name}.png"))

def load_manga_from_annotaion(img_path, anno_path, image_size):
    img = np.array(Image.open(img_path).convert("RGB"))
    with open(anno_path, 'r', encoding='utf-8') as f:
        annotation = json.load(f)
    # 
    ono_boxes = []
    original_ono_boxes = []
    # 
    offset = 50
    # 
    width = annotation['imageWidth']
    height = annotation['imageHeight']
    for shape in annotation['shapes']:
        pts = shape['points']
        label_name = shape['label']
        if label_name == 'Onomatopoeia-Kana':
            box = [
                int(max(min(pts[0][0], pts[1][0]) - offset, 0)),      # xmin
                int(max(min(pts[0][1], pts[1][1]) - offset, 0)),      # ymin
                int(min(max(pts[0][0], pts[1][0]) + offset, width)),  # xmax
                int(min(max(pts[0][1], pts[1][1]) + offset, height)), # ymax
            ]
            ono_boxes.append(box)
            box = [
                int(max(min(pts[0][0], pts[1][0]), 0)),      # xmin
                int(max(min(pts[0][1], pts[1][1]), 0)),      # ymin
                int(min(max(pts[0][0], pts[1][0]), width)),  # xmax
                int(min(max(pts[0][1], pts[1][1]), height)), # ymax
            ]
            original_ono_boxes.append(box)
    # 
    ono_images = []
    ono_recon_info = []
    for i, box in enumerate(ono_boxes):
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        crop_size = max(width, height)
        crop_img = img[ymin:ymax, xmin:xmax]

        anchor_x = 0
        anchor_y = 0
        if width != height:
            tmp_img = np.ones((crop_size, crop_size, 3), dtype=np.uint8) * 255
            if width > height:
                anchor_x = 0
                anchor_y = (width - height) // 2
            elif height > width:
                anchor_x = (height - width) // 2
                anchor_y = 0
            tmp_img[anchor_y:anchor_y+height, anchor_x:anchor_x+width] = crop_img 
            crop_img = tmp_img.copy()

        crop_img = cv2.resize(crop_img, (image_size, image_size))
        ono_images.append(crop_img)
        ono_recon_info.append([anchor_x, anchor_y, crop_size])   

    if len(ono_images) != 0:
        ono_images = [TF.to_tensor(x) for x in ono_images]
        ono_images = torch.stack(ono_images, dim=0)
        ono_boxes = torch.LongTensor(ono_boxes)
        original_ono_boxes = torch.LongTensor(original_ono_boxes)
        ono_recon_info = torch.LongTensor(ono_recon_info)
    else:
        ono_images = torch.tensor(ono_images)
        ono_recon_info = torch.tensor(ono_recon_info)
    return ono_images, ono_recon_info, ono_boxes, original_ono_boxes

def main_annotation(args, network, result_path, filter=None):
    network.cuda(args.gpu)
    network.eval()
    imgs_path = args.path
    annotation_path = args.anno_path
    with torch.no_grad():
        for manga in os.listdir(annotation_path):
            if filter is not None:
                if manga not in filter:
                    continue
            m_path = os.path.join(imgs_path, manga)
            a_path = os.path.join(annotation_path, manga)
            for epi in os.listdir(a_path):
                m_e_path = os.path.join(m_path, epi)
                a_e_path = os.path.join(a_path, epi)
                for chapter in os.listdir(a_e_path):
                    m_e_c_path = os.path.join(m_e_path, chapter)
                    a_e_c_path = os.path.join(a_e_path, chapter)
                    # 
                    origin_size_manga_folder = os.path.join(m_e_c_path, 'OriginSizeManga')
                    annotation_folder = os.path.join(a_e_c_path, 'annotation')
                    # 
                    tmp_output_dir = os.path.join(result_path, manga, epi, chapter, "OriginSizeOnomatopoeia")
                    makedirs(tmp_output_dir)
                    # 
                    for page_anno in os.listdir(annotation_folder):
                        try:
                            name = page_anno.split(".")[0]
                            img_path = os.path.join(origin_size_manga_folder, f"{name}.png")
                            anno_path = os.path.join(annotation_folder, page_anno)
                            ono_images, ono_recon_info, ono_boxes, original_ono_boxes = load_manga_from_annotaion(img_path, anno_path, args.img_size)
                            if ono_images.numel() != 0:
                                ono_images = ono_images.cuda(args.gpu)
                                preds = network(ono_images)
                                # save_test_batch(bubble_images, preds, result_path, f"test_{mname}_{epi}_{cha}_{name}")
                                paset_result_on_manga(
                                    img_path, 
                                    ono_recon_info, 
                                    ono_boxes, 
                                    preds, 
                                    tmp_output_dir, 
                                    f"{name.split('.')[0]}", 
                                    original_ono_boxes=original_ono_boxes
                                )
                        except:
                            print(name)
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--path", type=str, dest='path', default="D:/KungYao/Manga/MangaDatabase190926", help="Data path")
    parser.add_argument("--anno_path", type=str, dest='anno_path', default="D:/KungYao/Manga/MangaDatabase190926-extra-test", help="Data path")
    parser.add_argument("--model_path", type=str, dest='model_path', default=None, help="Model path")
    parser.add_argument('--img_size', type=int, dest='img_size', default=64)
    parser.add_argument('--gpu', type=int, dest='gpu', default=0)
    args = parser.parse_args()

    if args.model_path is None:
        raise ValueError("args.model_path should not be None.")
    obj = torch.load(args.model_path, map_location=f"cuda:{args.gpu}")
    net = ComposeNet(args.img_size)
    net.load_state_dict(obj["networks"]["G"].state_dict())
    res_output = "./results/manga/BE_font"
    makedirs(res_output)

    # main_mask(args, net, res_output, None) "DragonBall", 
    filter = ["AttackOnTitan", "InitialD", "DragonBall", "KurokosBasketball", "OnePiece"]
    # filter = ["DragonBall"]
    main_annotation(args, net, res_output, filter)

