import os
import argparse

import cv2
import torch
from torch._C import Value
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import DataLoader

from datasets .dataset import BCDataset
from models.networks_BC import ComposeNet
from tools.utils import makedirs

def point_filter(pts, img_size):
    h, w = img_size
    keep_idxs = []
    for i, pt in enumerate(pts):
        x, y = pt
        if x >= 0 and x < w and y >=0 and y < h:
            keep_idxs.append(i)
    keep_idxs = torch.LongTensor(keep_idxs)
    pts = pts[keep_idxs]
    return pts

# Only return imgs and bimgs.
def test_collate_fn(batch):
    imgs, bimgs, eimgs, cnt, key_cnt = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    bimgs = torch.stack(bimgs, dim=0)
    return imgs, bimgs
    
def viz_cnotours(contours, img_size):
    imgs = []
    h, w = img_size
    steps = torch.LongTensor([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])
    for cnt in contours:
        img = torch.zeros(img_size)
        if cnt.numel() != 0:
            # cnt = torch.LongTensor(point_filter(cnt))
            cnt = cnt.to(dtype=torch.long)
            img[cnt[:, 1], cnt[:, 0]] = 1
            # for pt in cnt:
                # for step in steps:
                #     step_pt = pt + step
                #     if step_pt[0] >=0 and step_pt[0] < w and step_pt[1] >=0 and step_pt[1] < h:
                #         img[step_pt[1], step_pt[0]] = 1
        imgs.append(img.reshape(1, h, w).repeat(3, 1, 1))
    imgs = torch.stack(imgs, dim=0)
    return imgs

def viz_resample_contours(contours, regressions, img_size):
    imgs = []
    h, w = img_size
    steps = torch.LongTensor([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])
    for i, cnt in enumerate(contours):
        pt_size = cnt.size(0)
        if pt_size != 0:
            regres_cnt = (cnt + regressions[i][:pt_size]).to(dtype=torch.long)
            cnt = cnt.to(dtype=torch.long)

            cnt_img = torch.zeros(img_size)
            cnt_img[cnt[:, 1], cnt[:, 0]] = 1

            relation_img = torch.zeros(img_size)
            relation_img = relation_img.numpy()
            for k, pt in enumerate(cnt):
                end_pt = regres_cnt[k]
                # if end_pt[0] >=0 and end_pt[0] < w and end_pt[1] >=0 and end_pt[1] < h:
                cv2.line(relation_img, (pt[0], pt[1]), (end_pt[0], end_pt[1]), (255, 255, 255), thickness=1)
            relation_img = TF.to_tensor(relation_img)
            relation_img = relation_img.squeeze()

            regres_cnt = point_filter(regres_cnt, img_size)
            regres_cnt_img = torch.zeros(img_size)
            regres_cnt_img[regres_cnt[:, 1], regres_cnt[:, 0]] = 1

            # TODO: draw mask by use of regress contours.
            imgs.append(torch.stack([cnt_img, relation_img, regres_cnt_img], dim=0))
        else:
            imgs.append(torch.zeros(3, h, w))
    imgs = torch.stack(imgs, dim=0)
    return imgs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--path", type=str, dest='path', default="D:/code/python/manga-python-tools/results/bbb-single-train-type", help="Data path")
    parser.add_argument("--model_path", type=str, dest='model_path', default=None, help="Model path")
    parser.add_argument('--img_size', type=int, dest='img_size', default=256)
    parser.add_argument('--gpu', type=int, dest='gpu', default=0)
    parser.add_argument('--batchsize', type=int, dest='batchsize', default=16)
    parser.add_argument('--debug', action="store_true", dest='debug')
    args = parser.parse_args()

    if args.debug:
        net = ComposeNet()
    else:
        if args.model_path is None:
            raise ValueError("args.model_path should not be None.")
        obj = torch.load(args.model_path, map_location=f"cuda:{args.gpu}")
        net = obj["networks"]
    padding = net.padding_for_contour
    res_output = "./results"
    makedirs(res_output)

    data_loader = DataLoader(
        BCDataset(args.path, (args.img_size, args.img_size), padding=padding, ifTest=True), 
        batch_size=args.batchsize, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=test_collate_fn, 
        pin_memory=True)

    net.cuda(args.gpu)
    net.eval()
    with torch.no_grad():
        for i, (imgs, _) in enumerate(data_loader):
            b = imgs.size(0)
            imgs = imgs.cuda(args.gpu)
            h, w = imgs.shape[-2:]
            size = torch.Size([h+padding*2, w+padding*2])

            preds = net(imgs)
            pred_edges = preds["edges"].cpu()
            pred_masks = preds["masks"].cpu()
            pred_cnts = preds["contours"]
            pred_regs = preds["contour_regressions"].cpu()

            imgs = F.pad(imgs.cpu(), (padding, padding, padding, padding), "constant", 0)
            pred_edges = F.pad(pred_edges.sigmoid(), (padding, padding, padding, padding), "constant", 0)
            pred_masks = F.pad(pred_masks.sigmoid(), (padding, padding, padding, padding), "constant", 0)
            # To 3 channels.
            pred_edges = pred_edges.repeat(1, 3, 1, 1)
            pred_masks = pred_masks.repeat(1, 3, 1, 1)
            # 
            img_contours = viz_cnotours(pred_cnts, size)
            img_contours_regress = viz_resample_contours(pred_cnts, pred_regs, size)

            vutils.save_image(
                torch.cat([imgs, pred_masks, pred_edges, img_contours, img_contours_regress], dim=0), 
                os.path.join(res_output, f"{i}.png"),
                nrow=b, 
                padding=2, 
                pad_value=1
            )
