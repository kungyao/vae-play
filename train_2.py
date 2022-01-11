import os
import argparse
from datetime import datetime

import torch

import torchvision.transforms.functional as TF
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.cuda.amp.autocast_mode import autocast

from tools.ops import *
from tools.utils import makedirs
from models.networks_2 import ComposeNet, find_tensor_contour, DEFAULT_MAX_POINTS
from datasets .dataset import BDataset

def train_collate_fn(batch):
    imgs, bimgs = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    bimgs = torch.stack(bimgs, dim=0)
    contours = find_tensor_contour(bimgs, max_points=1024)
    return imgs, bimgs, contours

def train(args, epoch, net, optim, train_loader):
    net.train()
    
    count = 0
    avg_loss = {
        "loss_mask": 0,
        "loss_regress": 0
    }

    bar = tqdm(train_loader)
    bar.set_description(f"epoch[{epoch}];")
    for i, (imgs, bimgs, contours) in enumerate(bar):

        # for x in range(imgs.size(0)):
        #     TF.to_pil_image(imgs[x]).save(os.path.join(args.res_output, f"{epoch}_{x}_a.png"))
        #     TF.to_pil_image(bimgs[x]).save(os.path.join(args.res_output, f"{epoch}_{x}_b.png"))

        imgs = imgs.cuda(args.gpu)
        bimgs = bimgs.cuda(args.gpu)
        # contours = [c.cuda(args.gpu) for c in contours]

        preds = net(imgs)
        pred_masks = preds["masks"]
        pred_cnts = preds["contours"]
        pred_regs = preds["contour_regressions"]

        loss_mask = 0.5 * F.binary_cross_entropy_with_logits(pred_masks, bimgs) + compute_dice_loss(pred_masks.sigmoid(), bimgs)
        loss_regress = compute_pt_regression_loss(pred_cnts, pred_regs, contours)
        losses = loss_mask + loss_regress

        optim.zero_grad()
        losses.backward()
        optim.step()
        
        next_count = count + imgs.size(0)
        avg_loss["loss_mask"] = (avg_loss["loss_mask"] * count + loss_mask.item()) / next_count
        avg_loss["loss_regress"] = (avg_loss["loss_regress"] * count + loss_regress.item()) / next_count
        count = next_count

        if (i+1) % args.viz_freq == 0:
            print("")
            res_str = ""
            for key in avg_loss:
                res_str += f"{key}: {round(avg_loss[key], 6)}; "
            print(res_str)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path', type=str, dest='path', default="D:/code/python/manga-python-tools/results/bbb-single-train-type")
    # 
    parser.add_argument('--lr', type=float, dest='lr', default=1e-4)
    parser.add_argument('--gpu', type=int, dest='gpu', default=0)
    parser.add_argument('--epoch', type=int, dest='epochs', default=20)
    parser.add_argument('--batchsize', type=int, dest='batchsize', default=32)
    #
    parser.add_argument('--workers', type=int, dest='workers', default=0)
    # 
    parser.add_argument('--img_size', type=int, dest='img_size', default=256)
    parser.add_argument('--max_points', type=int, dest='max_points', default=DEFAULT_MAX_POINTS)
    #
    parser.add_argument('--res_output', type=str, dest='res_output', default='./results')
    parser.add_argument('--model_output', type=str, dest='model_output', default='./logs')
    parser.add_argument('--viz_freq', type=int, dest='viz_freq', default=16)
    args = parser.parse_args()

    args.model_output = '{}_{}'.format(args.model_output, datetime.now().strftime("%Y%m%d-%H%M%S"))

    makedirs(args.res_output)
    makedirs(args.model_output)

    record_txt = open(os.path.join(args.model_output, "record.txt"), "w")
    for arg in vars(args):
        record_txt.write('{:35}{:20}\n'.format(arg, str(getattr(args, arg))))
    record_txt.close()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.img_size, args.img_size), interpolation=TF.InterpolationMode.NEAREST),
        # https://github.com/pytorch/vision/issues/9#issuecomment-304224800
        # Should make sure that img and target do the same operation.
        # 
        # transforms.RandomRotation(10, fill=0.0), 
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip()
    ])
    dset = BDataset(args.path, transform=transform, num_classes=4)
    dloader = DataLoader(
        dset, 
        batch_size=args.batchsize, 
        shuffle=True, 
        num_workers=args.workers, 
        collate_fn=train_collate_fn, 
        pin_memory=True)
    
    net = ComposeNet(max_points=args.max_points)
    net.cuda(args.gpu)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train(args, epoch, net, optim, dloader)
        if epoch > 10:
            torch.save(
                {
                    "networks": net, 
                    # "optims": optim,
                    "epoch": epoch
                }, 
                os.path.join(args.model_output, f"{epoch}.ckpt")
            )

