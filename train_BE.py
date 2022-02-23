import os
import argparse
from datetime import datetime

import torch
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from tqdm import tqdm, trange
from torchvision import transforms
from torch.utils.data import DataLoader

from datasets .dataset import BEDataset
from models.networks_BE import ComposeNet, Discriminator
from test_BE import save_test_batch
from tools.ops import *
from tools.utils import makedirs

def train_collate_fn(batch):
    imgs, bimgs, eimgs, labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    bimgs = torch.stack(bimgs, dim=0)
    eimgs = torch.stack(eimgs, dim=0)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    return imgs, bimgs, eimgs, labels

def train(args, epoch, iterations, nets, optims, train_loader):
    net = nets["NET"]
    disctiminator = nets["DISCRIMINATOR"]

    net.train()
    disctiminator.train()

    optim = optims["OPTIM"]
    disc_optim = optims["DISC_OPTIM"]
    
    count = 0
    avg_loss = {
        "loss_edge": 0,
        "loss_mask": 0, 
        "loss_adv": 0, 
        "loss_gen": 0
    }

    train_iter = iter(train_loader)
    for i in trange(iterations):
        try:
            imgs, bimgs, eimgs, labels = next(train_iter)
        except:
            train_iter = iter(train_loader)
            imgs, bimgs, eimgs, labels = next(train_iter)

        b = imgs.size(0)
        imgs = imgs.cuda(args.gpu)
        bimgs = bimgs.cuda(args.gpu)
        eimgs = eimgs.cuda(args.gpu)
        labels = labels.cuda(args.gpu)
        # contours = [c.cuda(args.gpu) for c in contours]

        valid = torch.ones(b, 1).cuda(args.gpu)
        fake = torch.zeros(b, 1).cuda(args.gpu)

        # with torch.no_grad():
        preds = net(imgs)
        pred_edges = preds["edges"]
        pred_masks = preds["masks"]
        
        # Train discriminator
        real_facticity, real_type = disctiminator(bimgs, eimgs, labels)
        fake_facticity, fake_type = disctiminator(pred_edges.detach(), pred_masks.detach(), labels)

        loss_adv_real = F.binary_cross_entropy(real_facticity, valid) + compute_hinge_loss(real_type, "d_real")
        loss_adv_fake = F.binary_cross_entropy(fake_facticity, fake) + compute_hinge_loss(fake_type, "d_fake")
        loss_adv = (loss_adv_real + loss_adv_fake) * 0.5

        disc_optim.zero_grad()
        loss_adv.backward()
        disc_optim.step()

        # Train (generator) ?
        # preds = net(imgs)
        # pred_edges = preds["edges"]
        # pred_masks = preds["masks"]
        g_facticity, g_type = disctiminator(pred_edges, pred_masks, labels)
        loss_egde = 0.5 * F.binary_cross_entropy_with_logits(pred_edges, eimgs) + compute_dice_loss(pred_edges.sigmoid(), eimgs)
        loss_mask = 0.5 * F.binary_cross_entropy_with_logits(pred_masks, bimgs) + compute_dice_loss(pred_masks.sigmoid(), bimgs)
        loss_gen = F.binary_cross_entropy(g_facticity, valid) + compute_hinge_loss(g_type, "g")
        losses = loss_egde + loss_mask + loss_gen * 0.1

        optim.zero_grad()
        losses.backward()
        optim.step()
        
        next_count = count + imgs.size(0)
        avg_loss["loss_edge"] = (avg_loss["loss_edge"] * count + loss_egde.item()) / next_count
        avg_loss["loss_mask"] = (avg_loss["loss_mask"] * count + loss_mask.item()) / next_count
        avg_loss["loss_adv"] = (avg_loss["loss_adv"] * count + loss_adv.item()) / next_count
        avg_loss["loss_gen"] = (avg_loss["loss_gen"] * count + loss_gen.item()) / next_count
        count = next_count

        if (i+1) % args.viz_freq == 0:
            print("")
            res_str = ""
            for key in avg_loss:
                res_str += f"{key}: {round(avg_loss[key], 6)}; "
            print(res_str)
            save_test_batch(imgs, preds, args.res_output, f"{epoch}_{i+1}")
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
    disc = Discriminator(1+2, args.img_size)

    net.cuda(args.gpu)
    disc.cuda(args.gpu)

    nets = {}
    nets["NET"] = net
    nets["DISCRIMINATOR"] = disc

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    disc_optim = torch.optim.Adam(disc.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, 10, gamma=0.5)

    optims = {}
    optims["OPTIM"] = optim
    optims["DISC_OPTIM"] = disc_optim

    for epoch in range(args.epochs):
        train(args, epoch, args.iterations, nets, optims, dloader)
        torch.save(
            {
                "networks": nets, 
                # "optims": optim,
                "epoch": epoch
            }, 
            os.path.join(args.model_output, f"{epoch}.ckpt")
        )
        # scheduler.step()

