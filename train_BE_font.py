# 
# Predict onomatopoeia mask.
# Comment D block for non-GAN training.
# 

import os
import argparse
from datetime import datetime

import torch
import torchvision.utils as vutils

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm, trange
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from datasets.dataset_font import FEDataset, AugmentOperator, prepare_syhthesis_data, ImageDataset
from models.networks_BE_font import ComposeNet, Discriminator
# from test_BE import save_test_batch
from tools.ops import *
from tools.utils import makedirs

def train(args, epoch, models, optims, base_loader, kana_loader, transform):
    print("--------------------------------------------------------------")
    net = models["net"]
    disc = models["disc"]

    optim = optims["net"]
    optim_style_enc = optims["style_enc"]
    optim_disc = optims["disc"]

    net.train()
    disc.train()

    count = 0
    avg_loss = {
        "loss_edge": 0,
        "loss_mask": 0, 
        # "loss_latent_label": 0, 
        # "loss_latent_style": 0, 
        "d_adv_real": 0, 
        "d_aux_real": 0, 
        "d_adv_fake": 0, 
        "loss_g_adv": 0, 
        "loss_g_aux": 0, 
        "loss_embed": 0, 
    }

    base_iter = iter(base_loader)
    for batch, (imgs, masks, labels) in enumerate(kana_loader):
        try:
            base_data = next(base_iter)
        except:
            base_iter = iter(base_loader)
            base_data = next(base_iter)

        base_img, base_target = base_data
        base_img = base_img[0]
        base_target = base_target[0]
        
        kana_imgs, kana_masks, kana_edge_masks, train_content_styles = prepare_syhthesis_data(base_img, base_target, imgs, masks, augmentor)
        
        labels = torch.LongTensor(labels)
        kana_imgs = [transform(kana_img) for kana_img in kana_imgs]
        kana_imgs = torch.stack(kana_imgs, dim=0)
        kana_masks = [transform(kana_mask) for kana_mask in kana_masks]
        kana_masks = torch.stack(kana_masks, dim=0)
        kana_edge_masks = [transform(kana_mask) for kana_mask in kana_edge_masks]
        kana_edge_masks = torch.stack(kana_edge_masks, dim=0)
        train_content_styles = torch.FloatTensor(train_content_styles)
        b = kana_imgs.size(0)

        # vutils.save_image(
        #     torch.cat([kana_imgs, kana_masks.repeat(1, 3, 1, 1), kana_edge_masks.repeat(1, 3, 1, 1)]), 
        #     os.path.join("./tmp", f'{epoch}_{batch}.png'),
        #     nrow=kana_imgs.size(0)
        # )
        # continue

        labels = labels.cuda(args.gpu)
        kana_imgs = kana_imgs.cuda(args.gpu)
        kana_masks = kana_masks.cuda(args.gpu)
        kana_edge_masks = kana_edge_masks.cuda(args.gpu)
        train_content_styles = train_content_styles.cuda(args.gpu)
        
        label_disc = torch.zeros((b, 143), dtype=train_content_styles.dtype)
        label_disc[torch.LongTensor(range(b)), labels] = 1
        label_disc = label_disc.cuda(args.gpu)
        train_y_map = {
            "cls": label_disc, 
            "cnt_style": train_content_styles
        }

        # Train D
        kana_gt_merge = torch.cat([kana_masks, kana_edge_masks], dim=1)
        with torch.no_grad():
            preds = net(kana_imgs, y=train_y_map)
            kana_pred_merge = torch.cat([preds["masks"].detach(), preds["edges"].detach()], dim=1)

        d_gt_adv, d_adv_aux = disc(kana_gt_merge, train_y_map) #  d_gt_cls_embed, d_gt_style_embed
        d_pred_adv, d_pred_aux = disc(kana_pred_merge, train_y_map) #  d_pred_cls_embed, d_pred_style_embed 

        optim_disc.zero_grad()
        d_adv_real = F.binary_cross_entropy(d_gt_adv, torch.ones((b, 1), device=d_gt_adv.device)) 
        d_aux_real = F.cross_entropy(d_adv_aux, labels)
        d_adv_fake = F.binary_cross_entropy(d_pred_adv, torch.zeros((b, 1), device=d_pred_adv.device)) # + F.cross_entropy(d_pred_cls, labels)
        d_adv_loss = (d_adv_real + d_adv_fake) * 0.5 + d_aux_real
        d_adv_loss.backward()
        optim_disc.step()

        # Train G 
        preds = net(kana_imgs, y=train_y_map)
        pred_masks = preds["masks"]
        pred_edges = preds["edges"]
        # pred_cls_embed = preds["y_cls"]
        # pred_style_embed = preds["y_cnt_style"]
        # pred_latent_label_cls = preds["latent_label_cls"]
        # pred_latent_style_cls = preds["latent_style_cls"]

        g_adv, g_aux = disc(torch.cat([pred_masks, pred_edges], dim=1), train_y_map)

        optim.zero_grad()
        loss_mask = 0.5 * F.binary_cross_entropy_with_logits(pred_masks, kana_masks) + compute_dice_loss(pred_masks.sigmoid(), kana_masks)
        loss_mask = loss_mask * 10
        # 
        loss_egde = 0.5 * F.binary_cross_entropy_with_logits(pred_edges, kana_edge_masks) + compute_dice_loss(pred_edges.sigmoid(), kana_edge_masks)
        loss_egde = loss_egde * 10
        # # 
        # loss_latent_label = F.cross_entropy(pred_latent_label_cls, labels)
        # loss_latent_label = loss_latent_label * 5.0
        # # 
        # loss_latent_style = F.cross_entropy(pred_latent_style_cls, train_content_styles)
        # loss_latent_style = loss_latent_style * 1.0
        # 
        loss_g_adv = F.binary_cross_entropy(g_adv, torch.ones((b, 1), device=g_adv.device))
        loss_g_adv = loss_g_adv * 2
        # 
        loss_g_aux = F.cross_entropy(g_aux, labels)
        loss_g_aux = loss_g_adv * 5
        # 
        ## + loss_latent_label + loss_latent_style
        losses = loss_egde + loss_mask + loss_g_adv + loss_g_aux
        losses.backward()
        optim.step()

        # 
        with torch.no_grad():
            preds = net(kana_imgs, y=train_y_map)
            pred_masks = preds["masks"]
            pred_edges = preds["edges"]
        preds = net(kana_imgs)
        pred_masks_ = preds["masks"]
        pred_edges_ = preds["edges"]

        optim_style_enc.zero_grad()
        # 
        loss_mask_ = 0.5 * F.binary_cross_entropy_with_logits(pred_masks_, kana_masks) + compute_dice_loss(pred_masks_.sigmoid(), kana_masks)
        loss_mask_ = loss_mask_ * 1.0
        # 
        loss_egde_ = 0.5 * F.binary_cross_entropy_with_logits(pred_edges_, kana_edge_masks) + compute_dice_loss(pred_edges_.sigmoid(), kana_edge_masks)
        loss_egde_ = loss_egde_ * 1.0
        # 
        loss_embed = F.l1_loss(pred_masks_, pred_masks) + F.l1_loss(pred_edges_, pred_edges)
        loss_embed = loss_embed * 2.0
        losses = loss_mask_ + loss_egde_ + loss_embed
        losses.backward()
        optim_style_enc.step()

        # with torch.no_grad():
        #     gt_cls_embed, gt_cnt_style_embed = net.embeding_block(labels, train_content_styles)
        # pred_cls_embed, pred_cnt_style_embed = net.style_encoder(kana_imgs, kana_imgs)
        # optim_style_enc.zero_grad()
        # loss_embed = F.l1_loss(pred_cls_embed, gt_cls_embed) + F.l1_loss(pred_cnt_style_embed, gt_cnt_style_embed)
        # loss_embed.backward()
        # optim_style_enc.step()
        
        next_count = count + kana_imgs.size(0)
        avg_loss["loss_edge"] = (avg_loss["loss_edge"] * count + loss_egde.item()) / next_count
        avg_loss["loss_mask"] = (avg_loss["loss_mask"] * count + loss_mask.item()) / next_count
        # avg_loss["loss_latent_label"] = (avg_loss["loss_latent_label"] * count + loss_latent_label.item()) / next_count
        # avg_loss["loss_latent_style"] = (avg_loss["loss_latent_style"] * count + loss_latent_style.item()) / next_count
        avg_loss["d_adv_real"] = (avg_loss["d_adv_real"] * count + d_adv_real.item()) / next_count
        avg_loss["d_aux_real"] = (avg_loss["d_aux_real"] * count + d_aux_real.item()) / next_count
        avg_loss["d_adv_fake"] = (avg_loss["d_adv_fake"] * count + d_adv_fake.item()) / next_count
        avg_loss["loss_g_adv"] = (avg_loss["loss_g_adv"] * count + loss_g_adv.item()) / next_count
        avg_loss["loss_g_aux"] = (avg_loss["loss_g_aux"] * count + loss_g_aux.item()) / next_count
        avg_loss["loss_embed"] = (avg_loss["loss_embed"] * count + loss_embed.item()) / next_count
        count = next_count

        if (batch+1) % args.viz_freq == 0:
            # print("")
            res_str = f"Epoch [{epoch}][{batch+1}]。"
            for key in avg_loss:
                res_str += f"{key}: {round(avg_loss[key], 6)}; "
            print(res_str)
            with torch.no_grad():
                preds = net(kana_imgs, y=train_y_map)
                pred_masks = preds["masks"]
                pred_edges = preds["edges"]
                preds = net(kana_imgs)
                pred_masks_ = preds["masks"]
                pred_edges_ = preds["edges"]
            vutils.save_image(
                torch.cat([
                    kana_imgs.cpu(), 
                    kana_masks.cpu().repeat(1, 3, 1, 1), 
                    pred_masks.cpu().repeat(1, 3, 1, 1),
                    pred_masks_.cpu().repeat(1, 3, 1, 1),
                    kana_edge_masks.cpu().repeat(1, 3, 1, 1),
                    pred_edges.cpu().repeat(1, 3, 1, 1), 
                    pred_edges_.cpu().repeat(1, 3, 1, 1)
                ]), 
                os.path.join(args.res_output, f'{epoch}_{batch}.png'),
                nrow=kana_imgs.size(0)
            )
            # save_test_batch(imgs, preds, args.res_output, f"{epoch}_{i+1}")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--path', type=str, dest='path', default="D:/Manga/bubble-gen-label")
    # 
    parser.add_argument('--lr', type=float, dest='lr', default=1e-4)
    parser.add_argument('--gpu', type=int, dest='gpu', default=0)
    parser.add_argument('--epoch', type=int, dest='epochs', default=1)
    # parser.add_argument('--iterations', type=int, dest='iterations', default=1000)
    parser.add_argument('--batchsize', type=int, dest='batchsize', default=32)
    #
    parser.add_argument('--workers', type=int, dest='workers', default=0)
    # 
    parser.add_argument('--img_size', type=int, dest='img_size', default=64)
    # parser.add_argument('--max_points', type=int, dest='max_points', default=DEFAULT_MAX_POINTS)
    #
    parser.add_argument('--res_output', type=str, dest='res_output', default='./results')
    parser.add_argument('--model_output', type=str, dest='model_output', default='./logs')
    parser.add_argument('--viz_freq', type=int, dest='viz_freq', default=20)
    args = parser.parse_args()

    dest_name = os.path.join("BE_font", datetime.now().strftime("%Y%m%d-%H%M%S"))
    args.res_output = os.path.join(args.res_output, dest_name)
    args.model_output = os.path.join(args.model_output, dest_name)

    makedirs(args.res_output)
    makedirs(args.model_output)

    record_txt = open(os.path.join(args.model_output, "record.txt"), "w")
    for arg in vars(args):
        record_txt.write('{:35}{:20}\n'.format(arg, str(getattr(args, arg))))
    record_txt.close()

    transform = T.Compose([
        T.Resize((args.img_size, args.img_size), interpolation=Image.BILINEAR), 
        T.ToTensor(),
    ])

    augmentor = AugmentOperator()
    # Datasets
    print("Load data.")
    base_dataset = ImageDataset("./training_data.json")
    print(f"base_dataset size is {len(base_dataset)}")
    kana_dataset = FEDataset()
    print(f"kana_dataset size is {len(kana_dataset)}")
    # Dataloader
    base_loader = DataLoader(base_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=ImageDataset.collate_fn)
    kana_batchsize = args.batchsize
    kana_loader = DataLoader(kana_dataset, batch_size=kana_batchsize, shuffle=True, num_workers=4, collate_fn=FEDataset.collate_fn)

    net = ComposeNet(args.img_size)
    disc = Discriminator(args.img_size, 2, 143)

    initialize_model(net)
    initialize_model(disc)

    net.cuda(args.gpu)
    disc.cuda(args.gpu)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    optim_style_enc = torch.optim.Adam(net.style_encoder.parameters(), lr=args.lr)
    optim_disc = torch.optim.Adam(disc.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, 10, gamma=0.5)

    models = {
        "net": net, 
        "disc": disc, 
    }

    optims = {
        "net": optim, 
        "style_enc": optim_style_enc, 
        "disc": optim_disc, 
    }

    for epoch in range(args.epochs):
        train(args, epoch, models, optims, base_loader, kana_loader, transform)
        torch.save(
            {
                "networks": models, 
                # "optims": optims,
                "epoch": epoch
            }, 
            os.path.join(args.model_output, f"{epoch}.ckpt")
        )
        # scheduler.step()

