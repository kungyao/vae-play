import os
import argparse
from datetime import datetime

import torch

import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from tqdm import tqdm, trange
from torchvision import transforms
from torch.utils.data import DataLoader

from tools.ops import *
from tools.utils import makedirs
from models.networks_BE import ComposeNet
from datasets .dataset import BEDataset

def train_collate_fn(batch):
    imgs, bimgs, eimgs = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    bimgs = torch.stack(bimgs, dim=0)
    eimgs = torch.stack(eimgs, dim=0)
    return imgs, bimgs, eimgs

def train(args, epoch, iterations, net, optim, train_loader):
    net.train()
    
    count = 0
    avg_loss = {
        "loss_edge": 0,
        "loss_mask": 0
    }

    train_iter = iter(train_loader)
    for i in trange(iterations):
        try:
            imgs, bimgs, eimgs = next(train_iter)
        except:
            train_iter = iter(train_loader)
            imgs, bimgs, eimgs = next(train_iter)

        imgs = imgs.cuda(args.gpu)
        bimgs = bimgs.cuda(args.gpu)
        eimgs = eimgs.cuda(args.gpu)
        # contours = [c.cuda(args.gpu) for c in contours]

        preds = net(imgs)
        pred_edges = preds["edges"]
        pred_masks = preds["masks"]

        loss_egde = 0.5 * F.binary_cross_entropy_with_logits(pred_edges, eimgs) + compute_dice_loss(pred_edges.sigmoid(), eimgs)
        loss_mask = 0.5 * F.binary_cross_entropy_with_logits(pred_masks, bimgs) + compute_dice_loss(pred_masks.sigmoid(), bimgs)
        losses = loss_egde + loss_mask

        optim.zero_grad()
        losses.backward()
        optim.step()
        
        next_count = count + imgs.size(0)
        avg_loss["loss_edge"] = (avg_loss["loss_edge"] * count + loss_egde.item()) / next_count
        avg_loss["loss_mask"] = (avg_loss["loss_mask"] * count + loss_mask.item()) / next_count
        count = next_count

        if (i+1) % args.viz_freq == 0:
            print("")
            res_str = ""
            for key in avg_loss:
                res_str += f"{key}: {round(avg_loss[key], 6)}; "
            print(res_str)
        
            b = imgs.size(0)
            imgs = imgs.cpu()
            pred_edges = pred_edges.cpu()
            pred_masks = pred_masks.cpu()

            pred_edges = pred_edges.repeat(1, 3, 1, 1)
            pred_masks = pred_masks.repeat(1, 3, 1, 1)

            vutils.save_image(
                torch.cat([imgs, pred_masks, pred_edges], dim=0), 
                os.path.join(args.res_output, f"{i+1}.png"),
                nrow=b, 
                padding=2, 
                pad_value=1
            )

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path', type=str, dest='path', default="D:/Manga/bubble-gen-label")
    # 
    parser.add_argument('--lr', type=float, dest='lr', default=1e-4)
    parser.add_argument('--gpu', type=int, dest='gpu', default=0)
    parser.add_argument('--epoch', type=int, dest='epochs', default=1)
    parser.add_argument('--iterations', type=int, dest='iterations', default=1000)
    parser.add_argument('--batchsize', type=int, dest='batchsize', default=32)
    #
    parser.add_argument('--workers', type=int, dest='workers', default=0)
    # 
    parser.add_argument('--img_size', type=int, dest='img_size', default=256)
    # parser.add_argument('--max_points', type=int, dest='max_points', default=DEFAULT_MAX_POINTS)
    #
    parser.add_argument('--res_output', type=str, dest='res_output', default='./results')
    parser.add_argument('--model_output', type=str, dest='model_output', default='./logs')
    parser.add_argument('--viz_freq', type=int, dest='viz_freq', default=100)
    args = parser.parse_args()

    args.model_output = os.path.join(args.model_output, "BE", datetime.now().strftime("%Y%m%d-%H%M%S"))

    makedirs(args.res_output)
    makedirs(args.model_output)

    record_txt = open(os.path.join(args.model_output, "record.txt"), "w")
    for arg in vars(args):
        record_txt.write('{:35}{:20}\n'.format(arg, str(getattr(args, arg))))
    record_txt.close()

    padding = 1
    # width height
    dset = BEDataset(args.path, (args.img_size, args.img_size))
    dloader = DataLoader(
        dset, 
        batch_size=args.batchsize, 
        shuffle=True, 
        num_workers=args.workers, 
        collate_fn=train_collate_fn, 
        pin_memory=True)
    
    net = ComposeNet()
    net.cuda(args.gpu)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 10, gamma=0.5)

    for epoch in range(args.epochs):
        train(args, epoch, args.iterations, net, optim, dloader)
        torch.save(
            {
                "networks": net, 
                # "optims": optim,
                "epoch": epoch
            }, 
            os.path.join(args.model_output, f"{epoch}.ckpt")
        )
        scheduler.step()

