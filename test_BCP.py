import os
import argparse

import cv2
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from datasets .dataset import BCPDatasetTEST
from models.networks_BCP import ComposeNet, VALUE_WEIGHT
from tools.utils import makedirs

def test_collate_fn_BCP(batch):
    imgs, bmasks = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    bmasks = torch.stack(bmasks, dim=0)
    return imgs, bmasks

def save_test_batch(imgs, bmasks, classes, contours, contour_targets, gt_annotation, result_path, result_name):
    b, c, h, w = imgs.shape
    
    results = []
    for i in range(b):
        tmp_b = bmasks[i].clone()
        tmp_b = torch.permute(tmp_b, (1, 2, 0))
        tmp_b = np.ascontiguousarray(tmp_b.numpy()).astype(np.uint8) * 255
        # Contour不用 /VALUE_WEIGHT
        cnt = (contours[i] * 0.5 + 0.5) * h
        # cnt_target = ((contour_targets[i] / VALUE_WEIGHT) * 0.5 + 0.5) * h
        cnt_target = ((contours[i] + contour_targets[i] / VALUE_WEIGHT) * 0.5 + 0.5) * h
        cnt_size = len(cnt)
        if classes[i] == 1:
            for j in range(cnt_size):
                pt = cnt[j].to(dtype=torch.long).tolist()
                end_pt = cnt_target[j].to(dtype=torch.long).tolist()
                cv2.line(tmp_b, (pt[0], pt[1]), (end_pt[0], end_pt[1]), (255, 255, 255), thickness=1)
        else:
            for j in range(cnt_size):
                pt = cnt_target[j].to(dtype=torch.long).tolist()
                end_pt = cnt_target[(j+1)%cnt_size].to(dtype=torch.long).tolist()
                cv2.line(tmp_b, (pt[0], pt[1]), (end_pt[0], end_pt[1]), (255, 255, 255), thickness=1)
        tmp_b = TF.to_tensor(tmp_b)
        results.append(tmp_b)
    results = torch.stack(results, dim=0)
    if gt_annotation is not None:
        gt_results = []
        contour_targets = [t["points"][:, 2:4].cpu() for t in gt_annotation]
        for i in range(b):
            tmp_b = bmasks[i].clone()
            tmp_b = torch.permute(tmp_b, (1, 2, 0))
            tmp_b = np.ascontiguousarray(tmp_b.numpy()).astype(np.uint8) * 255
            # Contour不用 /VALUE_WEIGHT
            cnt = (contours[i] * 0.5 + 0.5) * h
            # cnt_target = ((contour_targets[i] / VALUE_WEIGHT) * 0.5 + 0.5) * h
            cnt_target = ((contours[i] + contour_targets[i]) * 0.5 + 0.5) * h
            cnt_size = len(cnt)
            if classes[i] == 1:
                for j in range(cnt_size):
                    pt = cnt[j].to(dtype=torch.long).tolist()
                    end_pt = cnt_target[j].to(dtype=torch.long).tolist()
                    cv2.line(tmp_b, (pt[0], pt[1]), (end_pt[0], end_pt[1]), (255, 255, 255), thickness=1)
            else:
                for j in range(cnt_size):
                    pt = cnt_target[j].to(dtype=torch.long).tolist()
                    end_pt = cnt_target[(j+1)%cnt_size].to(dtype=torch.long).tolist()
                    cv2.line(tmp_b, (pt[0], pt[1]), (end_pt[0], end_pt[1]), (255, 255, 255), thickness=1)
            tmp_b = TF.to_tensor(tmp_b)
            gt_results.append(tmp_b)
        gt_results = torch.stack(gt_results, dim=0)
        results = torch.cat([imgs, gt_results, results], dim=0)
    else:
        results = torch.cat([imgs, results], dim=0)
    vutils.save_image(
        results, 
        os.path.join(result_path, f"{result_name}.png"),
        nrow=b, 
        padding=2, 
        pad_value=1
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--path", type=str, dest='path', default="../../python/manga-python-tools/results/ellipse", help="Data path")
    parser.add_argument("--model_path", type=str, dest='model_path', default=None, help="Model path")
    parser.add_argument('--img_size', type=int, dest='img_size', default=512)
    parser.add_argument('--gpu', type=int, dest='gpu', default=0)
    parser.add_argument('--batchsize', type=int, dest='batchsize', default=4)
    parser.add_argument('--debug', action="store_true", dest='debug')
    args = parser.parse_args()

    if args.debug:
        net = ComposeNet(args.img_size)
    else:
        if args.model_path is None:
            raise ValueError("args.model_path should not be None.")
        obj = torch.load(args.model_path, map_location=f"cuda:{args.gpu}")
        net = obj["networks"]
    res_output = "./results/BCP/"
    makedirs(res_output)

    # print(net.encoder.convs1[0].convs[0].conv[0].weight)
    # print(net.encoder.convs1[0].convs[0].conv[1].weight)

    net.cuda(args.gpu)
    net.eval()

    data_loader = DataLoader(
        BCPDatasetTEST("D:/Manga/bubble-gen-label", args.img_size), 
        batch_size=args.batchsize, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=test_collate_fn_BCP, 
        pin_memory=True)

    with torch.no_grad():
        for i, (imgs, bmasks) in enumerate(data_loader):
            imgs = imgs.cuda(args.gpu)
            preds = net(imgs)
            imgs = imgs.cpu()
            print(preds["classes"])
            _, classes = torch.max(preds["classes"], dim=1)
            contours = [x.cpu() for x in preds["contours"]]
            contour_targets = [x.cpu() for x in preds["target_pts"]]
            save_test_batch(imgs, bmasks, classes, contours, contour_targets, None, res_output, f"test_{i}")
            
