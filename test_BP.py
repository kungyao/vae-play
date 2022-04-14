import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import DataLoader

from datasets.dataset import BPDataset, BEDatasetGAN
from models.networks_BP import ComposeNet
from tools.utils import makedirs, rotate_vector, unit_vector

VALUE_WEIGHT = 10

def test_collate_fn_BP(batch):
    imgs, bmasks, ellipses, target = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    bmasks = torch.stack(bmasks, dim=0)
    ellipses = torch.stack(ellipses, dim=0)
    return imgs, bmasks, ellipses, target

def test_collate_fn_BEGAN(batch):
    imgs, bimgs, _ = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    bimgs = torch.stack(bimgs, dim=0)
    return imgs, bimgs

def save_target_batch(imgs, bmasks, ellipses, targets, result_path, result_name):
    b, c, h, w = imgs.shape

    if c == 3:
        imgs = imgs[:, 0, :, :].reshape(b, 1, h, w)
        ellipses = ellipses[:, 0, :, :].reshape(b, 1, h, w)

    results = []
    results_w_mask = []
    for i in range(b):
        tmp_img = torch.zeros((1, h, w))
        tmp_bmask = bmasks[i].clone()

        ellipse_param = targets[i]["phase1"]
        line_params = targets[i]["phase2"]

        # cx, cy, rx, ry, step = ellipse_param
        cx, cy, rx, ry = ellipse_param
        cx, cy, rx, ry = int((cx * 0.5 + 0.5) * w), int((cy * 0.5 + 0.5) * h), int(rx * 0.5 * w), int(ry * 0.5 * h)

        p_pt_xs = (line_params[:, 1] * 0.5 + 0.5) * w
        p_pt_ys = (line_params[:, 2] * 0.5 + 0.5) * h

        p_pt_end_xs = ((line_params[:, 1] + line_params[:, 3] * line_params[:, 5]) * 0.5 + 0.5) * w
        p_pt_end_ys = ((line_params[:, 2] + line_params[:, 4] * line_params[:, 5]) * 0.5 + 0.5) * h
        
        max_length = 256

        p_dpt_xs = (p_pt_end_xs - p_pt_xs) / max_length
        p_dpt_ys = (p_pt_end_ys - p_pt_ys) / max_length
        
        ray_sample = torch.arange(0, max_length, 1).reshape(1, -1).repeat(p_dpt_xs.size(0), 1)
        line_pt_xs = p_pt_xs.reshape(-1, 1).repeat(1, ray_sample.size(1)) + ray_sample * p_dpt_xs.reshape(-1, 1).repeat(1, ray_sample.size(1))
        line_pt_ys = p_pt_ys.reshape(-1, 1).repeat(1, ray_sample.size(1)) + ray_sample * p_dpt_ys.reshape(-1, 1).repeat(1, ray_sample.size(1))

        visual_pick = torch.logical_and(
                line_pt_xs>=0, torch.logical_and(
                    line_pt_xs<w, torch.logical_and(
                        line_pt_ys>=0, line_pt_ys<h)))
        
        line_pt_xs = line_pt_xs[visual_pick].to(dtype=torch.long)
        line_pt_ys = line_pt_ys[visual_pick].to(dtype=torch.long)

        tmp_img[0, line_pt_ys, line_pt_xs] = 1.0
        tmp_bmask[0, line_pt_ys, line_pt_xs] = 1.0
        
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

def save_test_batch(imgs, bmasks, ellipses, targets, predictions, result_path, result_name):
    b, c, h, w = imgs.shape
    # imgs = imgs[:, 0, :, :].reshape(b, 1, h, w)
    bmasks = bmasks[:, 0, :, :].reshape(b, 1, h, w)
    # ellipses = ellipses[:, 0, :, :].reshape(b, 1, h, w)

    pred_ellipse_params = predictions["ellipse_params"]
    # Weight
    pred_ellipse_params[:, :4] = pred_ellipse_params[:, :4] / VALUE_WEIGHT

    print("AAAAAAAAAAAAAAAAAA")
    print(pred_ellipse_params)
    print("BBBBBBBBBBBBBBBBBB")
    p1_targets = torch.stack([gt_target["phase1"] for gt_target in targets], dim=0)
    print(p1_targets)

    pred_triggers = predictions["if_triggers"]
    pred_line_params = predictions["line_params"]
    pred_sample_sample = predictions["sample_infos"]["sample"]

    results = []
    results_w_mask = []
    for i in range(b):
        tmp_img = torch.zeros((1, h, w))
        tmp_bmask = bmasks[i].clone()

        cx, cy, rx, ry, step = pred_ellipse_params[i].detach().cpu()
        cx, cy, rx, ry, step = int((cx * 0.5 + 0.5) * w), int((cy * 0.5 + 0.5) * h), int(rx * 0.5 * w), int(ry * 0.5 * h), int(step)
        # cx, cy, rx, ry = pred_ellipse_params[i].detach().cpu()
        # cx, cy, rx, ry = int((cx * 0.5 + 0.5) * w), int((cy * 0.5 + 0.5) * h), int(rx * 0.5 * w), int(ry * 0.5 * h)
        p_triggers = pred_triggers[i].detach().cpu()
        _, p_triggers = torch.max(p_triggers, dim=1)
        p_line_params = pred_line_params[i].detach().cpu()
        p_sample_sample = pred_sample_sample[i].detach().cpu()

        # Weight
        # p_line_params = p_line_params / VALUE_WEIGHT
        p_line_params[:, 0] = p_line_params[:, 0] / VALUE_WEIGHT
        p_line_params[:, 1] = p_line_params[:, 1] / VALUE_WEIGHT
        p_line_params[:, 3] = p_line_params[:, 3] / VALUE_WEIGHT

        p_pt_xs = ((p_sample_sample[:, 0] + p_line_params[:, 0]) * 0.5 + 0.5) * w
        p_pt_ys = ((p_sample_sample[:, 1] + p_line_params[:, 1]) * 0.5 + 0.5) * h

        # p_pt_end_xs = (p_line_params[:, 2] * 0.5 + 0.5) * w
        # p_pt_end_ys = (p_line_params[:, 3] * 0.5 + 0.5) * h
        
        thetas = p_line_params[:, 2]
        p_dpt_xs = p_sample_sample[:, 2]
        p_dpt_ys = p_sample_sample[:, 3]
        tmp_p_dpt_x = p_dpt_xs * torch.cos(thetas) - p_dpt_ys * torch.sin(thetas)
        tmp_p_dpt_y = p_dpt_xs * torch.sin(thetas) + p_dpt_ys * torch.cos(thetas)
        p_dpt_xs = tmp_p_dpt_x
        p_dpt_ys = tmp_p_dpt_y

        lengths = p_line_params[:, 3] * 0.5 * w
        lengths = lengths.to(dtype=torch.long)
        max_length = max(int(torch.max(lengths)), 1)

        # max_length = 128
        # p_dpt_xs = (p_pt_end_xs - p_pt_xs) / max_length
        # p_dpt_ys = (p_pt_end_ys - p_pt_ys) / max_length
        
        ray_sample = torch.arange(0, max_length, 1).reshape(1, -1).repeat(p_dpt_xs.size(0), 1)
        line_pt_xs = p_pt_xs.reshape(-1, 1).repeat(1, ray_sample.size(1)) + ray_sample * p_dpt_xs.reshape(-1, 1).repeat(1, ray_sample.size(1))
        line_pt_ys = p_pt_ys.reshape(-1, 1).repeat(1, ray_sample.size(1)) + ray_sample * p_dpt_ys.reshape(-1, 1).repeat(1, ray_sample.size(1))

        lengths = lengths.reshape(-1, 1).repeat(1, ray_sample.size(1))
        p_triggers = p_triggers.reshape(-1, 1).repeat(1, ray_sample.size(1))
        visual_pick = torch.logical_and(
            p_triggers==1, torch.logical_and(
                line_pt_xs>=0, torch.logical_and(
                    line_pt_xs<w, torch.logical_and(
                        line_pt_ys>=0, torch.logical_and(
                            line_pt_ys<h, torch.lt(ray_sample, lengths))))))
        # visual_pick = torch.logical_and(
        #     p_triggers==1, torch.logical_and(
        #         line_pt_xs>=0, torch.logical_and(
        #             line_pt_xs<w, torch.logical_and(
        #                 line_pt_ys>=0, line_pt_ys<h))))
        # visual_pick = torch.logical_and(
        #         line_pt_xs>=0, torch.logical_and(
        #             line_pt_xs<w, torch.logical_and(
        #                 line_pt_ys>=0, torch.logical_and(
        #                     line_pt_ys<h, torch.lt(ray_sample, lengths)))))
        
        line_pt_xs = line_pt_xs[visual_pick].to(dtype=torch.long)
        line_pt_ys = line_pt_ys[visual_pick].to(dtype=torch.long)

        tmp_img[0, line_pt_ys, line_pt_xs] = 1.0
        tmp_bmask[0, line_pt_ys, line_pt_xs] = 1.0
        
        for lx in range(rx):
            dst = cx + lx
            if dst >= 0 and dst < w:
                tmp_img[0, cy, dst] = 1.0
        for ly in range(ry):
            dst = cy + ly
            if dst >= 0 and dst < h:
                tmp_img[0, dst, cx] = 1.0

        results.append(tmp_img.repeat(3, 1, 1))
        results_w_mask.append(tmp_bmask.repeat(3, 1, 1))
    results = torch.stack(results, dim=0)
    results_w_mask = torch.stack(results_w_mask, dim=0)

    # bmasks, 
    vutils.save_image(
        torch.cat([imgs, results_w_mask, ellipses, results], dim=0), 
        os.path.join(result_path, f"{result_name}.png"),
        nrow=b, 
        padding=2, 
        pad_value=1
    )

def save_test_batch_(imgs, bmasks, predictions, result_path, result_name):
    b, c, h, w = imgs.shape

    if c == 3:
        imgs = imgs[:, 0, :, :].reshape(b, 1, h, w)
        bmasks = bmasks[:, 0, :, :].reshape(b, 1, h, w)

    pred_ellipse_params = predictions["ellipse_params"]
    # Weight
    pred_ellipse_params[:, :4] = pred_ellipse_params[:, :4] / VALUE_WEIGHT
    pred_triggers = predictions["if_triggers"]
    pred_line_params = predictions["line_params"]
    pred_sample_sample = predictions["sample_infos"]["sample"]

    results = []
    results_w_mask = []
    for i in range(b):
        tmp_img = torch.zeros((1, h, w))
        tmp_bmask = bmasks[i].clone()

        cx, cy, rx, ry, step = pred_ellipse_params[i].detach().cpu()
        cx, cy, rx, ry, step = int((cx * 0.5 + 0.5) * w), int((cy * 0.5 + 0.5) * h), int(rx * 0.5 * w), int(ry * 0.5 * h), int(step)
        # cx, cy, rx, ry = pred_ellipse_params[i].detach().cpu()
        # cx, cy, rx, ry = int((cx * 0.5 + 0.5) * w), int((cy * 0.5 + 0.5) * h), int(rx * 0.5 * w), int(ry * 0.5 * h)
        p_triggers = pred_triggers[i].detach().cpu()
        _, p_triggers = torch.max(p_triggers, dim=1)
        p_line_params = pred_line_params[i].detach().cpu()
        p_sample_sample = pred_sample_sample[i].detach().cpu()

        # Weight
        # p_line_params = p_line_params / VALUE_WEIGHT
        p_line_params[:, 0] = p_line_params[:, 0] / VALUE_WEIGHT
        p_line_params[:, 1] = p_line_params[:, 1] / VALUE_WEIGHT
        p_line_params[:, 3] = p_line_params[:, 3] / VALUE_WEIGHT

        p_pt_xs = ((p_sample_sample[:, 0] + p_line_params[:, 0]) * 0.5 + 0.5) * w
        p_pt_ys = ((p_sample_sample[:, 1] + p_line_params[:, 1]) * 0.5 + 0.5) * h

        # p_pt_end_xs = (p_line_params[:, 2] * 0.5 + 0.5) * w
        # p_pt_end_ys = (p_line_params[:, 3] * 0.5 + 0.5) * h
        
        thetas = p_line_params[:, 2]
        p_dpt_xs = p_sample_sample[:, 2]
        p_dpt_ys = p_sample_sample[:, 3]
        tmp_p_dpt_x = p_dpt_xs * torch.cos(thetas) - p_dpt_ys * torch.sin(thetas)
        tmp_p_dpt_y = p_dpt_xs * torch.sin(thetas) + p_dpt_ys * torch.cos(thetas)
        p_dpt_xs = tmp_p_dpt_x
        p_dpt_ys = tmp_p_dpt_y

        lengths = p_line_params[:, 3] * 0.5 * w
        lengths = lengths.to(dtype=torch.long)
        max_length = max(int(torch.max(lengths)), 1)

        # max_length = 128
        # p_dpt_xs = (p_pt_end_xs - p_pt_xs) / max_length
        # p_dpt_ys = (p_pt_end_ys - p_pt_ys) / max_length
        
        ray_sample = torch.arange(0, max_length, 1).reshape(1, -1).repeat(p_dpt_xs.size(0), 1)
        line_pt_xs = p_pt_xs.reshape(-1, 1).repeat(1, ray_sample.size(1)) + ray_sample * p_dpt_xs.reshape(-1, 1).repeat(1, ray_sample.size(1))
        line_pt_ys = p_pt_ys.reshape(-1, 1).repeat(1, ray_sample.size(1)) + ray_sample * p_dpt_ys.reshape(-1, 1).repeat(1, ray_sample.size(1))

        lengths = lengths.reshape(-1, 1).repeat(1, ray_sample.size(1))
        p_triggers = p_triggers.reshape(-1, 1).repeat(1, ray_sample.size(1))
        visual_pick = torch.logical_and(
            p_triggers==1, torch.logical_and(
                line_pt_xs>=0, torch.logical_and(
                    line_pt_xs<w, torch.logical_and(
                        line_pt_ys>=0, torch.logical_and(
                            line_pt_ys<h, torch.lt(ray_sample, lengths))))))
        
        line_pt_xs = line_pt_xs[visual_pick].to(dtype=torch.long)
        line_pt_ys = line_pt_ys[visual_pick].to(dtype=torch.long)

        tmp_img[0, line_pt_ys, line_pt_xs] = 1.0
        tmp_bmask[0, line_pt_ys, line_pt_xs] = 1.0
        
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
        torch.cat([imgs, results_w_mask, results], dim=0), 
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

    net.cuda(args.gpu)
    net.eval()

    # data_loader = DataLoader(
    #     BPDataset(args.path, args.img_size), 
    #     batch_size=args.batchsize, 
    #     shuffle=False, 
    #     num_workers=4, 
    #     collate_fn=test_collate_fn_BP, 
    #     pin_memory=True)
    
    # with torch.no_grad():
    #     for i, (imgs, bmasks, ellipses, targets) in enumerate(data_loader):
    #         imgs = imgs.cuda(args.gpu)
    #         ellipses = ellipses.cuda(args.gpu)
    #         preds = net(imgs)
    #         # preds = net(ellipses)
    #         imgs = imgs.cpu()
    #         ellipses = ellipses.cpu()
    #         # save_target_batch(imgs, bmasks, ellipses, targets, res_output, f"target_{i}")
    #         save_test_batch(imgs, bmasks, ellipses, targets, preds, res_output, f"test_{i}")

    data_loader = DataLoader(
        BEDatasetGAN("D:/Manga/bubble-gen-label", (args.img_size, args.img_size), select_list=["3"]), 
        batch_size=args.batchsize, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=test_collate_fn_BEGAN, 
        pin_memory=True)

    with torch.no_grad():
        for i, (imgs, bmasks) in enumerate(data_loader):
            imgs = imgs.cuda(args.gpu)
            preds = net(imgs)
            imgs = imgs.cpu()
            save_test_batch_(imgs, bmasks, preds, res_output, f"test_{i}")
        
        
