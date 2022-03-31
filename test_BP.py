import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import DataLoader

from datasets.dataset import BPDataset
from models.networks_BP import ComposeNet
from tools.utils import makedirs, rotate_vector, unit_vector

def test_collate_fn(batch):
    imgs, bmasks, ellipses, target = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    bmasks = torch.stack(bmasks, dim=0)
    ellipses = torch.stack(ellipses, dim=0)
    return imgs, bmasks, ellipses, target

def save_test_batch(imgs, bmasks, ellipses, targets, predictions, result_path, result_name):
    b, c, h, w = imgs.shape

    if c == 3:
        imgs = imgs[:, 0, :, :].reshape(b, 1, h, w)
        ellipses = ellipses[:, 0, :, :].reshape(b, 1, h, w)

    print("AAAAAAAAAAAAAAAAAA")
    pred_ellipse_params = predictions["ellipse_params"]
    print(pred_ellipse_params)
    # Weight
    pred_ellipse_params[:, :4] = pred_ellipse_params[:, :4]/10
    # print(pred_ellipse_params)

    pred_triggers = predictions["if_triggers"]
    pred_line_params = predictions["line_params"]
    pred_sample_sample = predictions["sample_infos"]["sample"]


    results = []
    results_w_mask = []
    for i in range(b):
        tmp_img = torch.zeros((1, h, w))
        tmp_bmask = bmasks[i].clone()

        # cx, cy, rx, ry, step = pred_ellipse_params[i].detach().cpu()
        # cx, cy, rx, ry, step = int((cx * 0.5 + 0.5) * w), int((cy * 0.5 + 0.5) * h), int(rx * w), int(ry * h), int(step)
        cx, cy, rx, ry = pred_ellipse_params[i].detach().cpu()
        cx, cy, rx, ry = int((cx * 0.5 + 0.5) * w), int((cy * 0.5 + 0.5) * h), int(rx * w), int(ry * h)
        p_triggers = pred_triggers[i].detach().cpu()
        p_line_params = pred_line_params[i].detach().cpu()
        p_sample_sample = pred_sample_sample[i].detach().cpu()

        p_pt_xs = ((p_sample_sample[:, 0] + p_line_params[:, 0]) * 0.5 + 0.5) * w
        p_pt_ys = ((p_sample_sample[:, 1] + p_line_params[:, 1]) * 0.5 + 0.5) * h
        
        thetas = p_line_params[:, 2]
        p_dpt_xs = p_sample_sample[:, 2]
        p_dpt_ys = p_sample_sample[:, 3]
        tmp_p_dpt_x = p_dpt_xs * torch.cos(thetas) - p_dpt_ys * torch.sin(thetas)
        tmp_p_dpt_y = p_dpt_xs * torch.sin(thetas) + p_dpt_ys * torch.cos(thetas)
        p_dpt_xs = tmp_p_dpt_x
        p_dpt_ys = tmp_p_dpt_y

        lengths = p_line_params[:, 3] * w
        lengths = lengths.to(dtype=torch.long)
        max_length = max(int(torch.max(lengths)), 1)
        
        ray_sample = torch.arange(0, max_length, 1).reshape(1, -1).repeat(p_dpt_xs.size(0), 1)
        line_pt_xs = p_pt_xs.reshape(-1, 1).repeat(1, ray_sample.size(1)) + ray_sample * p_dpt_xs.reshape(-1, 1).repeat(1, ray_sample.size(1))
        line_pt_ys = p_pt_ys.reshape(-1, 1).repeat(1, ray_sample.size(1)) + ray_sample * p_dpt_ys.reshape(-1, 1).repeat(1, ray_sample.size(1))

        lengths = lengths.reshape(-1, 1).repeat(1, ray_sample.size(1))
        p_triggers = p_triggers.reshape(-1, 1).repeat(1, ray_sample.size(1))
        print(torch.max(p_triggers), torch.min(p_triggers))
        visual_pick = torch.logical_and(
            p_triggers>0.2, torch.logical_and(
                line_pt_xs>=0, torch.logical_and(
                    line_pt_xs<w, torch.logical_and(
                        line_pt_ys>=0, torch.logical_and(
                            line_pt_ys<h, torch.lt(ray_sample, lengths))))))
        # visual_pick = torch.logical_and(
        #         line_pt_xs>=0, torch.logical_and(
        #             line_pt_xs<w, torch.logical_and(
        #                 line_pt_ys>=0, torch.logical_and(
        #                     line_pt_ys<h, torch.lt(ray_sample, lengths)))))
        
        line_pt_xs = line_pt_xs[visual_pick].to(dtype=torch.long)
        line_pt_ys = line_pt_ys[visual_pick].to(dtype=torch.long)

        tmp_img[0, line_pt_ys, line_pt_xs] = 1.0
        tmp_bmask[0, line_pt_ys, line_pt_xs] = 1.0

        # for j in range(len(p_triggers)):
        #     if p_triggers[j] > 0.01:
        #         p_offset_x = p_line_params[j][0]
        #         p_offset_y = p_line_params[j][1]
        #         p_theta = p_line_params[j][2]
        #         p_length = max(0, min(p_line_params[j][3] * w, w - 1)) # or h

        #         p_pt_x = ((p_sample_sample[j][0] + p_offset_x) * 0.5 + 0.5) * w
        #         p_pt_y = ((p_sample_sample[j][1] + p_offset_y) * 0.5 + 0.5) * h
        #         # p_pt_x = p_sample_sample[j][0] + p_offset_x
        #         # p_pt_y = p_sample_sample[j][1] + p_offset_y
        #         p_dpt_x = p_sample_sample[j][2]
        #         p_dpt_y = p_sample_sample[j][3]

        #         p_dpt_x, p_dpt_y = rotate_vector(p_dpt_x, p_dpt_y, p_theta)
        #         line_pt_x = p_pt_x + torch.arange(0, p_length, 0.5) * p_dpt_x
        #         line_pt_y = p_pt_y + torch.arange(0, p_length, 0.5) * p_dpt_y

        #         line_pt_x = line_pt_x.to(dtype=torch.long)
        #         line_pt_y = line_pt_y.to(dtype=torch.long)

        #         for pt_x, pt_y in zip(line_pt_x, line_pt_y):
        #             if pt_x >=0 and pt_x < w and pt_y >=0 and pt_y < h:
        #                 tmp_img[0, pt_y, pt_x] = 1.0
        #                 tmp_bmask[0, pt_y, pt_x] = 1.0
        
        for lx in range(rx):
            dst = cx + lx
            if dst >= 0 and dst < w:
                tmp_img[0, cy, dst] = 1.0
        for ly in range(ry):
            dst = cy + ly
            if dst >= 0 and dst < h:
                tmp_img[0, dst, cx] = 1.0

        results.append(tmp_img)
        results_w_mask.append(tmp_bmask)
    results = torch.stack(results, dim=0)
    results_w_mask = torch.stack(results_w_mask, dim=0)

    # bmasks, 
    vutils.save_image(
        torch.cat([imgs, ellipses, results, results_w_mask], dim=0), 
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
    res_output = "./results"
    makedirs(res_output)

    data_loader = DataLoader(
        BPDataset(args.path, args.img_size), 
        batch_size=args.batchsize, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=test_collate_fn, 
        pin_memory=True)

    net.cuda(args.gpu)
    net.eval()
    with torch.no_grad():
        for i, (imgs, bmasks, ellipses, targets) in enumerate(data_loader):
            imgs = imgs.cuda(args.gpu)
            ellipses = ellipses.cuda(args.gpu)
            preds = net(ellipses)
            imgs = imgs.cpu()
            ellipses = ellipses.cpu()
            save_test_batch(imgs, bmasks, ellipses, targets, preds, res_output, f"test_{i}")
            break
