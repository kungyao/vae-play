import os
import argparse
from datetime import datetime

import torch
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from tqdm import tqdm, trange
from torch.utils.data import DataLoader

from test_BE import save_test_batch

from datasets.dataset import BEGanDataset
from models.networks_BE_GAN import *
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
    G = nets["G"]
    D = nets["D"]

    g_opt = optims["G"]
    d_opt = optims["D"]

    G.train()
    D.train()
    
    count = 0
    avg_loss = {
        "d_adv_loss": 0,
        "d_type_loss": 0,
        "loss_edge": 0,
        "loss_mask": 0,
        "g_adv_loss": 0,
        "g_type_loss": 0
    }

    train_iter = iter(train_loader)
    for i in trange(iterations):
        try:
            imgs, bimgs, eimgs, labels = next(train_iter)
        except:
            train_iter = iter(train_loader)
            imgs, bimgs, eimgs, labels = next(train_iter)

        b = imgs.size(0)
        # Prepare data
        imgs = imgs.cuda(args.gpu)
        # Bubble only mask
        bimgs = bimgs.cuda(args.gpu)
        # Boundary only image
        eimgs = eimgs.cuda(args.gpu)
        labels = labels.cuda(args.gpu)

        # D
        with torch.no_grad():
            preds = G(imgs)
            pred_masks = preds["masks"]
            pred_edges = preds["edges"]
        d_real_type, d_real_feats = D(imgs, bimgs, eimgs)
        d_fake_type, d_fake_feats = D(imgs, pred_masks, pred_edges)

        d_adv_loss = 1 - torch.mean(torch.abs(d_fake_feats - d_real_feats))
        d_type_loss = F.cross_entropy(d_real_type, labels)
        d_losses = d_adv_loss + d_type_loss

        d_opt.zero_grad()
        d_losses.backward()
        d_opt.step()

        # G - GT
        preds = G(imgs)
        pred_masks = preds["masks"]
        pred_edges = preds["edges"]

        with torch.no_grad():
            _, g_real_feats = D(imgs, bimgs, eimgs)
        g_pred_type, g_pred_feats = D(imgs, pred_masks, pred_edges)

        loss_mask = 0.5 * F.binary_cross_entropy_with_logits(pred_masks, bimgs) + compute_dice_loss(pred_masks.sigmoid(), bimgs)
        loss_egde = 0.5 * F.binary_cross_entropy_with_logits(pred_edges, eimgs) + compute_dice_loss(pred_edges.sigmoid(), eimgs)
        g_adv_loss = torch.mean(torch.abs(g_pred_feats - g_real_feats))
        g_type_loss = F.cross_entropy(g_pred_type, labels)
        losses = loss_mask + loss_egde + g_adv_loss + g_type_loss

        g_opt.zero_grad()
        losses.backward()
        g_opt.step()
        # 
        next_count = count + imgs.size(0)
        avg_loss["d_adv_loss"] = (avg_loss["d_adv_loss"] * count + d_adv_loss.item()) / next_count
        avg_loss["d_type_loss"] = (avg_loss["d_type_loss"] * count + d_type_loss.item()) / next_count
        avg_loss["loss_edge"] = (avg_loss["loss_edge"] * count + loss_egde.item()) / next_count
        avg_loss["loss_mask"] = (avg_loss["loss_mask"] * count + loss_mask.item()) / next_count
        avg_loss["g_adv_loss"] = (avg_loss["g_adv_loss"] * count + g_adv_loss.item()) / next_count
        avg_loss["g_type_loss"] = (avg_loss["g_type_loss"] * count + g_adv_loss.item()) / next_count
        count = next_count

        if (i+1) % args.viz_freq == 0:
            print("")
            res_str = f"[Epoch: {epoch}]。"
            for key in avg_loss:
                res_str += f"{key}: {round(avg_loss[key], 6)}; "
            print(res_str)
            with torch.no_grad():
                # with gt mask
                preds = G(imgs)
                save_test_batch(imgs, preds, args.res_output, f"{epoch}_{i+1}_wgtm")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path', type=str, dest='path', default="D:/Manga/bubble-gen-label")
    # 
    parser.add_argument('--lr', type=float, dest='lr', default=1e-4)
    parser.add_argument('--gpu', type=int, dest='gpu', default=0)
    parser.add_argument('--epochs', type=int, dest='epochs', default=10)
    parser.add_argument('--iterations', type=int, dest='iterations', default=200)
    parser.add_argument('--batchsize', type=int, dest='batchsize', default=16)
    #
    parser.add_argument('--workers', type=int, dest='workers', default=0)
    # 
    parser.add_argument('--z_size', type=int, dest='z_size', default=32, help="Final image size for encoder.")
    parser.add_argument('--img_size', type=int, dest='img_size', default=512)
    #
    parser.add_argument('--res_output', type=str, dest='res_output', default='./results')
    parser.add_argument('--model_output', type=str, dest='model_output', default='./logs')
    parser.add_argument('--viz_freq', type=int, dest='viz_freq', default=20)
    args = parser.parse_args()

    dest_name = os.path.join("BE_GAN", datetime.now().strftime("%Y%m%d-%H%M%S"))
    args.res_output = os.path.join(args.res_output, dest_name)
    args.model_output = os.path.join(args.model_output, dest_name)

    makedirs(args.res_output)
    makedirs(args.model_output)

    record_txt = open(os.path.join(args.model_output, "record.txt"), "w")
    for arg in vars(args):
        record_txt.write('{:35}{:20}\n'.format(arg, str(getattr(args, arg))))
    record_txt.close()

    generator = ComposeNet(3, args.img_size)
    # unknown, oval, explode, emit
    discriminator = Discriminator(3, args.img_size, 4)

    initialize_model(generator)
    initialize_model(discriminator)

    nets = {}
    nets["G"] = generator
    nets["D"] = discriminator

    optims = {}
    optims["G"] = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optims["D"] = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    for name, net in nets.items():
        nets[name] = net.cuda(args.gpu)

    dset = BEGanDataset(args.path, args.img_size, if_test=False)
    dloader = DataLoader(
        dset, 
        batch_size=args.batchsize, 
        shuffle=True, 
        num_workers=args.workers, 
        collate_fn=train_collate_fn, 
        pin_memory=True)

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

