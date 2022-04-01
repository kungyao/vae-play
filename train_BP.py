import os
import argparse
from datetime import datetime

import torch
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from tqdm import tqdm, trange
from torchvision import transforms
from torch.utils.data import DataLoader

from datasets .dataset import BPDataset
from models.networks_BP import ComposeNet
from test_BP import save_test_batch
from tools.ops import *
from tools.utils import makedirs, rotate_vector

def train_collate_fn(batch):
    imgs, bmasks, ellipses, target = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    bmasks = torch.stack(bmasks, dim=0)
    ellipses = torch.stack(ellipses, dim=0)
    return imgs, bmasks, ellipses, target

def train(args, epoch, iterations, net, optim, train_loader):
    net.train()
    
    count = 0
    avg_loss = {
        "loss_ellipse_param": 0,
        "loss_emit_line_param": 0
    }

    train_iter = iter(train_loader)
    for i in trange(iterations):
        try:
            imgs, bmasks, ellipses, targets = next(train_iter)
        except:
            train_iter = iter(train_loader)
            imgs, bmasks, ellipses, targets = next(train_iter)

        b = imgs.size(0)
        imgs = imgs.cuda(args.gpu)
        # bmasks = bmasks.cuda(args.gpu)
        ellipses = ellipses.cuda(args.gpu)
        # targets = [{k: v.cuda(args.gpu) for k, v in t.items()} for t in targets]

        preds = net(ellipses)
        pred_ellipse_params = preds["ellipse_params"]

        p1_targets = torch.stack([gt_target["phase1"] for gt_target in targets], dim=0)
        loss_ellipse_param = compute_ellipse_param_loss(pred_ellipse_params, p1_targets)

        p2_targets = torch.stack([gt_target["phase2"] for gt_target in targets], dim=0)
        loss_emit_line_param = compute_ellipse_pt_loss(preds, p2_targets)

        losses = loss_ellipse_param + loss_emit_line_param

        optim.zero_grad()
        losses.backward()
        optim.step()
        
        with torch.no_grad():
            next_count = count + imgs.size(0)
            avg_loss["loss_ellipse_param"] = (avg_loss["loss_ellipse_param"] * count + loss_ellipse_param.item()) / next_count
            avg_loss["loss_emit_line_param"] = (avg_loss["loss_emit_line_param"] * count + loss_emit_line_param.item()) / next_count
            count = next_count

            if (i+1) % args.viz_freq == 0:
                print("")
                res_str = ""
                for key in avg_loss:
                    res_str += f"{key}: {round(avg_loss[key], 6)}; "
                print(res_str)
                imgs = imgs.cpu()
                ellipses = ellipses.cpu()
                save_test_batch(imgs, bmasks, ellipses, targets, preds, args.res_output, f"{epoch}_{i+1}")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path', type=str, dest='path', default="../../python/manga-python-tools/results/ellipse")
    # 
    parser.add_argument('--lr', type=float, dest='lr', default=1e-3)
    parser.add_argument('--gpu', type=int, dest='gpu', default=0)
    parser.add_argument('--epoch', type=int, dest='epochs', default=1)
    parser.add_argument('--iterations', type=int, dest='iterations', default=500)
    parser.add_argument('--batchsize', type=int, dest='batchsize', default=8)
    #
    parser.add_argument('--workers', type=int, dest='workers', default=0)
    # 
    parser.add_argument('--img_size', type=int, dest='img_size', default=512)
    # parser.add_argument('--max_points', type=int, dest='max_points', default=DEFAULT_MAX_POINTS)
    #
    parser.add_argument('--res_output', type=str, dest='res_output', default='./results')
    parser.add_argument('--model_output', type=str, dest='model_output', default='./logs')
    parser.add_argument('--viz_freq', type=int, dest='viz_freq', default=10)
    args = parser.parse_args()

    dest_name = os.path.join("BP", datetime.now().strftime("%Y%m%d-%H%M%S"))
    args.res_output = os.path.join(args.res_output, dest_name)
    args.model_output = os.path.join(args.model_output, dest_name)

    makedirs(args.res_output)
    makedirs(args.model_output)

    record_txt = open(os.path.join(args.model_output, "record.txt"), "w")
    for arg in vars(args):
        record_txt.write('{:35}{:20}\n'.format(arg, str(getattr(args, arg))))
    record_txt.close()

    # width height
    dset = BPDataset(args.path, args.img_size)
    dloader = DataLoader(
        dset, 
        batch_size=args.batchsize, 
        shuffle=True, 
        num_workers=args.workers, 
        collate_fn=train_collate_fn, 
        pin_memory=True)
    
    net = ComposeNet(args.img_size)
    initialize_model(net.encoder)
    initialize_model(net.ellipse_predictor)
    initialize_model(net.emit_line_predictor)

    net.cuda(args.gpu)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    step_size = args.epochs // 3
    step_size = 1 if step_size == 0 else step_size
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size, gamma=0.1)

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
