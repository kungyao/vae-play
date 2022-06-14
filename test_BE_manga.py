import os
import argparse

import cv2
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

def paset_result_on_manga(img_path, bubble_recon_info, bubble_masks, bubble_labels, bubble_boxes, predictions, result_path, result_name):
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

    merge_edge_mask = torch.zeros(h, w)
    merge_bubble_mask = torch.zeros(h, w)
    for i in range(b):
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
            tmp_mask = bubble_masks[i][0, ymin:ymax, xmin:xmax]
            dilate_mask = torch.clamp(torch.nn.functional.conv2d(tmp_mask[None][None], kernel_tensor, padding=(padding, padding)), 0, 1).squeeze()
            merge_edge_mask[ymin:ymax, xmin:xmax] = dilate_mask - tmp_mask
            merge_bubble_mask[ymin:ymax, xmin:xmax] = tmp_mask

    merge_edge_mask = merge_edge_mask.to(dtype=torch.bool)
    merge_bubble_mask = merge_bubble_mask.to(dtype=torch.bool)
    # # 
    # img[:, merge_edge_mask] = img[:, merge_edge_mask] // 2
    # img[0, merge_edge_mask] = img[0, merge_edge_mask] + 255 // 2
    # # 
    # img[:, merge_bubble_mask] = img[:, merge_bubble_mask] // 2
    # img[1, merge_bubble_mask] = img[1, merge_bubble_mask] + 255 // 2
    # img = img.to(dtype=torch.float) / 255
    # vutils.save_image(img, os.path.join("./results", f"{result_name}.png"))

    merge_edge_mask = merge_edge_mask.to(dtype=torch.float)
    vutils.save_image(merge_edge_mask, os.path.join(result_path, f"{result_name}.png"))
    

def load_manga(img_path, mask_path, bimage_size):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--path", type=str, dest='path', default="D:/KungYao/Manga/MangaDatabase190926", help="Data path")
    parser.add_argument("--model_path", type=str, dest='model_path', default=None, help="Model path")
    parser.add_argument('--img_size', type=int, dest='img_size', default=256)
    parser.add_argument('--gpu', type=int, dest='gpu', default=0)
    args = parser.parse_args()

    if args.model_path is None:
        raise ValueError("args.model_path should not be None.")
    obj = torch.load(args.model_path, map_location=f"cuda:{args.gpu}")
    net = ComposeNet()
    net.load_state_dict(obj["networks"].state_dict())
    res_output = "./results/manga"
    makedirs(res_output)

    net.cuda(args.gpu)
    net.eval()
    with torch.no_grad():
        for mname in os.listdir(args.path):
            # if mname != "ShyuraNoMon":
            #     continue
            mdir = os.path.join(args.path, mname)
            for epi in os.listdir(mdir):
                edir = os.path.join(mdir, epi)
                for cha in os.listdir(edir):
                    cdir = os.path.join(edir, cha)
                    #
                    idir = os.path.join(cdir, "OriginSizeManga")
                    bidir = os.path.join(cdir, "OriginSizeBubbles")
                    if os.path.exists(idir) and os.path.exists(bidir):
                        tmp_output_dir = os.path.join(res_output, mname, epi, cha, "OriginSizeBubbleEdges")
                        makedirs(tmp_output_dir)
                        for name in os.listdir(idir):
                            # if name != "ShyuraNoMon_3_10_076.png":
                            #     continue
                            img_path = os.path.join(idir, name)
                            mask_path = os.path.join(bidir, name)
                            if os.path.exists(mask_path):
                                bubble_images, bubble_recon_info, bubble_masks, bubble_labels, bubble_boxes = load_manga(img_path, mask_path, args.img_size)
                                if bubble_images.numel() != 0:
                                    bubble_images = bubble_images.cuda(args.gpu)
                                    preds = net(bubble_images)
                                    # save_test_batch(bubble_images, preds, res_output, f"test_{mname}_{epi}_{cha}_{name}")
                                    paset_result_on_manga(
                                        img_path, 
                                        bubble_recon_info, 
                                        bubble_masks, 
                                        bubble_labels, 
                                        bubble_boxes, 
                                        preds, 
                                        tmp_output_dir, 
                                        f"{name.split('.')[0]}"
                                    )

