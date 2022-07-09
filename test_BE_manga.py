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

from models.networks_BE import ComposeNet
from tools.utils import makedirs

BUBBLE_TYPES = {
    "Oval": 1, 
    "Explosion": 2, 
    "NoFrame": 3,
    "Box": 4,
}

def save_test_batch(imgs, predictions, result_path, result_name):
    b = imgs.size(0)
    pred_edges = predictions["edges"].cpu()
    pred_masks = predictions["masks"].cpu()
    imgs = (imgs.cpu() * 255).to(dtype=torch.uint8)

    pred_edges = pred_edges.cpu()
    pred_masks = pred_masks.cpu()

    pred_edges[pred_edges>=0.5] = 1
    pred_edges[pred_edges<0.5] = 0

    pred_masks[pred_masks>=0.5] = 1
    pred_masks[pred_masks<0.5] = 0

    pred_edges = pred_edges.to(dtype=torch.bool)
    pred_masks = pred_masks.to(dtype=torch.bool)

    result_edges = []
    result_bubbles = []
    for j in range(b):
        result_edges.append(vutils.draw_segmentation_masks(imgs[j], masks=pred_edges[j], alpha=0.5, colors=(255, 0, 0)))
        result_bubbles.append(vutils.draw_segmentation_masks(imgs[j], masks=pred_masks[j], alpha=0.5, colors=(255, 0, 0)))
    result_edges = torch.stack(result_edges, dim=0).to(dtype=torch.float) / 255
    result_bubbles = torch.stack(result_bubbles, dim=0).to(dtype=torch.float) / 255

    # To 3 channels.
    # pred_edges = pred_edges.repeat(1, 3, 1, 1)
    pred_masks = pred_masks.repeat(1, 3, 1, 1)

    vutils.save_image(
        torch.cat([result_edges, result_bubbles, pred_masks], dim=0), 
        os.path.join(result_path, f"{result_name}.png"),
        nrow=b, 
        padding=2, 
        pad_value=1
    )

def paset_result_on_manga(img_path, bubble_recon_info, bubble_masks, bubble_labels, bubble_boxes, predictions, result_path, result_name, original_bubble_boxes=None):
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

    kernel_size = 13
    padding = (kernel_size - 1) // 2
    kernel = torch.ones(kernel_size, kernel_size)
    kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0)) # size: (1, 1, 3, 3)

    # 0~255
    result_mask = torch.zeros(3, h, w, dtype=torch.uint8)
    # Check if occupied
    check_mask = torch.zeros(h, w, dtype=torch.bool)
    for i in range(b):
        # Create new mask for part saving
        merge_edge_mask = torch.zeros(h, w)
        merge_bubble_mask = torch.zeros(h, w)
        anchor_x, anchor_y, size = bubble_recon_info[i].tolist()
        xmin, ymin, xmax, ymax = bubble_boxes[i].tolist()
        width = xmax - xmin
        height = ymax - ymin
        if bubble_labels[i] != 3:
            tmp_edge = pred_edges[i]
            tmp_edge = F.interpolate(tmp_edge[None], size=(size, size), mode='nearest')[0]
            merge_edge_mask[ymin:ymax, xmin:xmax] = tmp_edge[0, anchor_y:anchor_y+height, anchor_x:anchor_x+width]
            # merge_edge_mask = merge_edge_mask.to(dtype=torch.bool)

            tmp_mask = pred_masks[i]
            tmp_mask = F.interpolate(tmp_mask[None], size=(size, size), mode='nearest')[0]
            merge_bubble_mask[ymin:ymax, xmin:xmax] = tmp_mask[0, anchor_y:anchor_y+height, anchor_x:anchor_x+width]
            # merge_bubble_mask = merge_bubble_mask.to(dtype=torch.bool)
        else:
            if original_bubble_boxes is None:
                tmp_mask = bubble_masks[i][0, ymin:ymax, xmin:xmax]
                dilate_mask = torch.clamp(torch.nn.functional.conv2d(tmp_mask[None][None], kernel_tensor, padding=(padding, padding)), 0, 1).squeeze()
                merge_edge_mask[ymin:ymax, xmin:xmax] = dilate_mask - tmp_mask
                merge_bubble_mask[ymin:ymax, xmin:xmax] = tmp_mask
            else:
                oxmin, oymin, oxmax, oymax = original_bubble_boxes[i].tolist()
                oxmin = anchor_x + oxmin - xmin
                oymin = anchor_y + oymin - ymin
                oxmax = anchor_x + oxmax - xmin
                oymax = anchor_y + oymax - ymin
                tmp_mask = torch.zeros(size, size, dtype=pred_masks[i].dtype)
                tmp_mask[oymin:oymax, oxmin:oxmax] = 1.0
                dilate_mask = torch.clamp(torch.nn.functional.conv2d(tmp_mask[None][None], kernel_tensor, padding=(padding, padding)), 0, 1).squeeze()
                merge_edge_mask[ymin:ymax, xmin:xmax] = (dilate_mask - tmp_mask)[anchor_y:anchor_y+height, anchor_x:anchor_x+width]
                merge_bubble_mask[ymin:ymax, xmin:xmax] = tmp_mask[anchor_y:anchor_y+height, anchor_x:anchor_x+width]
        # 
        merge_edge_mask = merge_edge_mask.to(dtype=torch.bool)
        merge_bubble_mask = merge_bubble_mask.to(dtype=torch.bool)
        # Preserve Content Region and Remove Content Region From Edge Mask
        merge_edge_mask = merge_edge_mask * ~merge_bubble_mask
        # Remove Occupied Area
        merge_edge_mask = merge_edge_mask * ~check_mask
        merge_bubble_mask = merge_bubble_mask * ~check_mask
        # 
        toatl_merge_mask = merge_edge_mask + merge_bubble_mask
        check_mask = check_mask + toatl_merge_mask
        # To float
        merge_edge_mask = merge_edge_mask.to(dtype=torch.uint8) * 255
        merge_bubble_mask = merge_bubble_mask.to(dtype=torch.uint8) * 255
        toatl_merge_mask = toatl_merge_mask.to(dtype=torch.uint8) * bubble_labels[i]
        # RGB
        # toatl_merge_mask = torch.stack([merge_bubble_mask, toatl_merge_mask, merge_edge_mask], dim=0)
        # BGR
        toatl_merge_mask = torch.stack([merge_edge_mask, toatl_merge_mask, merge_bubble_mask], dim=0)
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

def paset_edge_result_on_manga(img_path, bubble_recon_info, bubble_masks, bubble_labels, bubble_boxes, predictions, result_path, result_name, original_bubble_boxes=None):
    pred_edges = predictions["edges"].cpu()
    b = pred_edges.size(0)

    pred_edges = pred_edges.cpu()

    pred_edges[pred_edges>=0.5] = 1
    pred_edges[pred_edges<0.5] = 0

    img = TF.to_tensor(Image.open(img_path).convert("RGB")) * 255
    img = img.to(dtype=torch.uint8)
    c, h, w = img.shape

    kernel_size = 13
    padding = (kernel_size - 1) // 2
    kernel = torch.ones(kernel_size, kernel_size)
    kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0)) # size: (1, 1, 3, 3)

    # 0~255
    result_mask = torch.zeros(3, h, w, dtype=torch.uint8)
    # Check if occupied
    check_mask = torch.zeros(h, w, dtype=torch.bool)
    for i in range(b):
        # Create new mask for part saving
        merge_edge_mask = torch.zeros(h, w)
        merge_bubble_mask = torch.zeros(h, w)
        anchor_x, anchor_y, size = bubble_recon_info[i].tolist()
        xmin, ymin, xmax, ymax = bubble_boxes[i].tolist()
        width = xmax - xmin
        height = ymax - ymin
        if bubble_labels[i] != 3:
            tmp_edge = pred_edges[i]
            tmp_edge = F.interpolate(tmp_edge[None], size=(size, size), mode='nearest')[0]
            merge_edge_mask[ymin:ymax, xmin:xmax] = tmp_edge[0, anchor_y:anchor_y+height, anchor_x:anchor_x+width]
            # merge_edge_mask = merge_edge_mask.to(dtype=torch.bool)

            merge_bubble_mask[ymin:ymax, xmin:xmax] = bubble_masks[i][0, ymin:ymax, xmin:xmax]
            # merge_bubble_mask = merge_bubble_mask.to(dtype=torch.bool)
        else:
            tmp_mask = bubble_masks[i][0, ymin:ymax, xmin:xmax]
            dilate_mask = torch.clamp(torch.nn.functional.conv2d(tmp_mask[None][None], kernel_tensor, padding=(padding, padding)), 0, 1).squeeze()
            merge_edge_mask[ymin:ymax, xmin:xmax] = dilate_mask - tmp_mask
            merge_bubble_mask[ymin:ymax, xmin:xmax] = tmp_mask
        # 
        merge_edge_mask = merge_edge_mask.to(dtype=torch.bool)
        merge_bubble_mask = merge_bubble_mask.to(dtype=torch.bool)
        # Preserve Content Region and Remove Content Region From Edge Mask
        merge_edge_mask = merge_edge_mask * ~merge_bubble_mask
        # Remove Occupied Area
        merge_edge_mask = merge_edge_mask * ~check_mask
        merge_bubble_mask = merge_bubble_mask * ~check_mask
        # 
        toatl_merge_mask = merge_edge_mask + merge_bubble_mask
        check_mask = check_mask + toatl_merge_mask
        # To float
        merge_edge_mask = merge_edge_mask.to(dtype=torch.uint8) * 255
        merge_bubble_mask = merge_bubble_mask.to(dtype=torch.uint8) * 255
        toatl_merge_mask = toatl_merge_mask.to(dtype=torch.uint8) * bubble_labels[i]
        # RGB
        # toatl_merge_mask = torch.stack([merge_bubble_mask, toatl_merge_mask, merge_edge_mask], dim=0)
        # BGR
        toatl_merge_mask = torch.stack([merge_edge_mask, toatl_merge_mask, merge_bubble_mask], dim=0)
        result_mask = result_mask + toatl_merge_mask
    result_mask[:, ~check_mask] = 255
    result_mask = result_mask.permute(1, 2, 0).numpy()
    cv2.imwrite(os.path.join(result_path, f"{result_name}.png"), result_mask)

def load_manga_from_mask(img_path, mask_path, bimage_size):
    img = np.array(Image.open(img_path).convert("RGB"))
    bubble_mask = np.array(Image.open(mask_path).convert("RGB"))
    # Extract only ''red'' pixel.
    white = np.where((bubble_mask[:,:,0]==255) & (bubble_mask[:,:,1]==255) & (bubble_mask[:,:,2]==255))
    h, w = bubble_mask.shape[:2]
    bubble_mask[white] = (0, 0, 0)
    bubble_label_mask = bubble_mask[:,:,1]
    bubble_mask = bubble_mask[:,:,0]
    labeled, ncomponents = scipy_label(bubble_mask)
    bubble_boxes = []
    bubble_masks = []
    bubble_labels = []
    # obj_ids = np.arange(1, ncomponents+1)
    # print(self.imgs[index], ncomponents)
    for n in range(ncomponents):
        mask = (labeled == (n + 1)).astype(np.uint8)
        pos = np.where(mask)
        xmin = max(np.min(pos[1]) - 200, 0)
        ymin = max(np.min(pos[0]) - 200, 0)
        xmax = min(np.max(pos[1]) + 200, w - 1)
        ymax = min(np.max(pos[0]) + 200, h - 1)
        bubble_boxes.append([xmin, ymin, xmax, ymax])
        bubble_masks.append(mask)
        # Get pixel label from image
        bubble_labels.append(bubble_label_mask[pos][0])
    
    bubble_images = []
    bubble_recon_info = []
    for i, box in enumerate(bubble_boxes):
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

        crop_img = cv2.resize(crop_img, (bimage_size, bimage_size))
        bubble_images.append(crop_img)
        bubble_recon_info.append([anchor_x, anchor_y, crop_size])

    if len(bubble_images) != 0:
        bubble_images = [TF.to_tensor(x) for x in bubble_images]
        bubble_images = torch.stack(bubble_images, dim=0)
        bubble_masks = [TF.to_tensor(x * 255) for x in bubble_masks]
        bubble_masks = torch.stack(bubble_masks, dim=0)
        bubble_boxes = torch.LongTensor(bubble_boxes)
        bubble_labels = torch.LongTensor(bubble_labels)
        bubble_recon_info = torch.LongTensor(bubble_recon_info)
    else:
        bubble_images = torch.tensor(bubble_images)
        bubble_recon_info = torch.tensor(bubble_recon_info)
    return bubble_images, bubble_recon_info, bubble_masks, bubble_labels, bubble_boxes

def load_manga_from_annotaion(img_path, anno_path, bimage_size):
    img = np.array(Image.open(img_path).convert("RGB"))
    with open(anno_path, 'r', encoding='utf-8') as f:
        annotation = json.load(f)
    # 
    bubble_boxes = []
    original_bubble_boxes = []
    bubble_masks = []
    bubble_labels = []
    # 
    offset = 50
    # 
    width = annotation['imageWidth']
    height = annotation['imageHeight']
    for shape in annotation['shapes']:
        pts = shape['points']
        label_name = shape['label']
        if label_name == 'Bubble-Boundary':
            box = [
                int(max(min(pts[0][0], pts[1][0]) - offset, 0)),      # xmin
                int(max(min(pts[0][1], pts[1][1]) - offset, 0)),      # ymin
                int(min(max(pts[0][0], pts[1][0]) + offset, width)),  # xmax
                int(min(max(pts[0][1], pts[1][1]) + offset, height)), # ymax
            ]
            bubble_boxes.append(box)
            box = [
                int(max(min(pts[0][0], pts[1][0]), 0)),      # xmin
                int(max(min(pts[0][1], pts[1][1]), 0)),      # ymin
                int(min(max(pts[0][0], pts[1][0]), width)),  # xmax
                int(min(max(pts[0][1], pts[1][1]), height)), # ymax
            ]
            original_bubble_boxes.append(box)
            sub_label_name = shape['sub_label'] if 'sub_label' in shape else None
            tmp_label = -1
            if sub_label_name is not None:
                if sub_label_name in BUBBLE_TYPES:
                    tmp_label = BUBBLE_TYPES[sub_label_name]
            bubble_labels.append(tmp_label)
            bubble_masks.append(torch.zeros(0))
    # 
    bubble_images = []
    bubble_recon_info = []
    for i, box in enumerate(bubble_boxes):
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

        crop_img = cv2.resize(crop_img, (bimage_size, bimage_size))
        bubble_images.append(crop_img)
        bubble_recon_info.append([anchor_x, anchor_y, crop_size])   

    if len(bubble_images) != 0:
        bubble_images = [TF.to_tensor(x) for x in bubble_images]
        bubble_images = torch.stack(bubble_images, dim=0)
        # bubble_masks = [TF.to_tensor(x * 255) for x in bubble_masks]
        bubble_masks = torch.stack(bubble_masks, dim=0)
        bubble_boxes = torch.LongTensor(bubble_boxes)
        original_bubble_boxes = torch.LongTensor(original_bubble_boxes)
        bubble_labels = torch.LongTensor(bubble_labels)
        bubble_recon_info = torch.LongTensor(bubble_recon_info)
    else:
        bubble_images = torch.tensor(bubble_images)
        bubble_recon_info = torch.tensor(bubble_recon_info)
    return bubble_images, bubble_recon_info, bubble_masks, bubble_labels, bubble_boxes, original_bubble_boxes

def main_mask(args, network, result_path, filter=None):
    network.cuda(args.gpu)
    network.eval()
    with torch.no_grad():
        for mname in os.listdir(args.path):
            if filter is not None:
                if mname not in filter:
                    continue
            mdir = os.path.join(args.path, mname)
            for epi in os.listdir(mdir):
                edir = os.path.join(mdir, epi)
                for cha in os.listdir(edir):
                    cdir = os.path.join(edir, cha)
                    #
                    idir = os.path.join(cdir, "OriginSizeManga")
                    bidir = os.path.join(cdir, "OriginSizeBubbles")
                    if not os.path.exists(idir) or not os.path.exists(bidir):
                        continue
                    tmp_output_dir = os.path.join(result_path, mname, epi, cha, "OriginSizeBubbleEdges")
                    makedirs(tmp_output_dir)
                    for name in os.listdir(idir):
                        img_path = os.path.join(idir, name)
                        mask_path = os.path.join(bidir, name)
                        if os.path.exists(mask_path):
                            bubble_images, bubble_recon_info, bubble_masks, bubble_labels, bubble_boxes = load_manga_from_mask(img_path, mask_path, args.img_size)
                            if bubble_images.numel() != 0:
                                bubble_images = bubble_images.cuda(args.gpu)
                                preds = network(bubble_images)
                                # save_test_batch(bubble_images, preds, result_path, f"test_{mname}_{epi}_{cha}_{name}")
                                paset_edge_result_on_manga(
                                    img_path, 
                                    bubble_recon_info, 
                                    bubble_masks, 
                                    bubble_labels, 
                                    bubble_boxes, 
                                    preds, 
                                    tmp_output_dir, 
                                    f"{name.split('.')[0]}"
                                )
    return 

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
                    tmp_output_dir = os.path.join(result_path, manga, epi, chapter, "OriginSizeBubbleEdges")
                    makedirs(tmp_output_dir)
                    # 
                    for page_anno in os.listdir(annotation_folder):
                        try:
                            name = page_anno.split(".")[0]
                            img_path = os.path.join(origin_size_manga_folder, f"{name}.png")
                            anno_path = os.path.join(annotation_folder, page_anno)
                            bubble_images, bubble_recon_info, bubble_masks, bubble_labels, bubble_boxes, original_bubble_boxes = load_manga_from_annotaion(img_path, anno_path, args.img_size)
                            if bubble_images.numel() != 0:
                                bubble_images = bubble_images.cuda(args.gpu)
                                preds = network(bubble_images)
                                # save_test_batch(bubble_images, preds, result_path, f"test_{mname}_{epi}_{cha}_{name}")
                                paset_result_on_manga(
                                    img_path, 
                                    bubble_recon_info, 
                                    bubble_masks, 
                                    bubble_labels, 
                                    bubble_boxes, 
                                    preds, 
                                    tmp_output_dir, 
                                    f"{name.split('.')[0]}", 
                                    original_bubble_boxes=original_bubble_boxes
                                )
                        except:
                            print(name)
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--path", type=str, dest='path', default="D:/KungYao/Manga/MangaDatabase190926", help="Data path")
    parser.add_argument("--anno_path", type=str, dest='anno_path', default="D:/KungYao/Manga/MangaDatabase190926-extra-test", help="Data path")
    parser.add_argument("--model_path", type=str, dest='model_path', default=None, help="Model path")
    parser.add_argument('--img_size', type=int, dest='img_size', default=512)
    parser.add_argument('--gpu', type=int, dest='gpu', default=0)
    args = parser.parse_args()

    if args.model_path is None:
        raise ValueError("args.model_path should not be None.")
    obj = torch.load(args.model_path, map_location=f"cuda:{args.gpu}")
    net = ComposeNet()
    net.load_state_dict(obj["networks"].state_dict())
    res_output = "./results/manga"
    makedirs(res_output)

    # main_mask(args, net, res_output, None) "DragonBall", 
    filter = ["AttackOnTitan", "InitialD", "KurokosBasketball", "OnePiece"]
    # filter = ["DragonBall"]
    main_annotation(args, net, res_output, filter)

