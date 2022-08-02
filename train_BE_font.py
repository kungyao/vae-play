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
from test_BE import save_test_batch
from tools.ops import *
from tools.utils import makedirs

WITH_DISC = True

def train(args, epoch, nets, optims, base_loader, kana_loader, transform):
    print("--------------------------------------------------------------")
    G = nets["G"]
    g_opt = optims["G"]
    G.train()

    if WITH_DISC:
        D = nets["D"]
        d_opt = optims["D"]
        D.train()
    
    count = 0
    avg_loss = {
        "loss_edge": 0,
        "loss_mask": 0,
    }

    if WITH_DISC:
        avg_loss["d_adv_loss"] = 0
        avg_loss["d_type_loss"] = 0
        avg_loss["g_adv_loss"] = 0
        avg_loss["g_type_loss"] = 0

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
        
        # label_disc = torch.zeros((b, 143), dtype=train_content_styles.dtype)
        # label_disc[torch.LongTensor(range(b)), labels] = 1
        # label_disc = label_disc.cuda(args.gpu)
        # train_y_map = {
        #     "cls": label_disc, 
        #     "cnt_style": train_content_styles
        # }

        if WITH_DISC:
            # D
            with torch.no_grad():
                preds = G(kana_imgs)
                pred_masks = preds["masks"].sigmoid()
                pred_edges = preds["edges"].sigmoid()
            # x_cls, x_param, feats
            d_real_type, d_real_param, d_real_feats = D(kana_imgs, kana_masks, kana_edge_masks)
            d_fake_type, d_fake_param, d_fake_feats = D(kana_imgs, pred_masks, pred_edges)

            d_adv_loss = 1 - torch.mean(torch.abs(d_fake_feats - d_real_feats))
            d_type_loss = F.cross_entropy(d_real_type, labels) + (1 - torch.mean(torch.abs(d_fake_type - d_real_type))) * 0.2
            d_param_loss = F.l1_loss(d_real_param, train_content_styles) + (1 - torch.mean(torch.abs(d_fake_param - d_real_param))) * 0.2
            d_losses = d_adv_loss + d_type_loss + d_param_loss

            d_opt.zero_grad()
            d_losses.backward()
            d_opt.step()

        # G - GT
        preds = G(kana_imgs)
        pred_masks = preds["masks"]
        pred_edges = preds["edges"]

        if WITH_DISC:
            with torch.no_grad():
                g_real_type, g_real_param, g_real_feats = D(kana_imgs, kana_masks, kana_edge_masks)
            g_pred_type, g_pred_param, g_pred_feats = D(kana_imgs, pred_masks.sigmoid(), pred_edges.sigmoid())

        loss_mask = 0.5 * F.binary_cross_entropy_with_logits(pred_masks, kana_masks) + compute_dice_loss(pred_masks.sigmoid(), kana_masks)
        loss_egde = 0.5 * F.binary_cross_entropy_with_logits(pred_edges, kana_edge_masks) + compute_dice_loss(pred_edges.sigmoid(), kana_edge_masks)
        losses = loss_mask * 2 + loss_egde * 2

        if WITH_DISC:
            g_adv_loss = torch.mean(torch.abs(g_pred_feats - g_real_feats))
            g_type_loss = torch.mean(torch.abs(g_pred_type - g_real_type)) # F.cross_entropy(g_pred_type, labels)
            g_param_loss = torch.mean(torch.abs(g_pred_param - g_real_param)) # F.cross_entropy(g_pred_type, labels)
            losses = losses + g_adv_loss + g_type_loss + g_param_loss

        g_opt.zero_grad()
        losses.backward()
        g_opt.step()
        # 
        next_count = count + kana_imgs.size(0)
        avg_loss["loss_edge"] = (avg_loss["loss_edge"] * count + loss_egde.item()) / next_count
        avg_loss["loss_mask"] = (avg_loss["loss_mask"] * count + loss_mask.item()) / next_count
        if WITH_DISC:
            avg_loss["d_adv_loss"] = (avg_loss["d_adv_loss"] * count + d_adv_loss.item()) / next_count
            avg_loss["d_type_loss"] = (avg_loss["d_type_loss"] * count + d_type_loss.item()) / next_count
            avg_loss["g_adv_loss"] = (avg_loss["g_adv_loss"] * count + g_adv_loss.item()) / next_count
            avg_loss["g_type_loss"] = (avg_loss["g_type_loss"] * count + g_type_loss.item()) / next_count
        count = next_count

        if (batch+1) % args.viz_freq == 0:
            print("")
            res_str = f"[Epoch: {epoch}。{batch}]。"
            for key in avg_loss:
                res_str += f"{key}: {round(avg_loss[key], 6)}; "
            print(res_str)
            with torch.no_grad():
                # with gt mask
                preds = G(kana_imgs)
                save_test_batch(kana_imgs, preds, args.res_output, f"{epoch}_{batch+1}")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--path', type=str, dest='path', default="D:/Manga/bubble-gen-label")
    # 
    parser.add_argument('--lr', type=float, dest='lr', default=1e-4)
    parser.add_argument('--gpu', type=int, dest='gpu', default=0)
    parser.add_argument('--epoch', type=int, dest='epochs', default=20)
    parser.add_argument('--batchsize', type=int, dest='batchsize', default=32)
    #
    parser.add_argument('--workers', type=int, dest='workers', default=0)
    # 
    parser.add_argument('--img_size', type=int, dest='img_size', default=64)
    #
    parser.add_argument('--res_output', type=str, dest='res_output', default='./results')
    parser.add_argument('--model_output', type=str, dest='model_output', default='./logs')
    parser.add_argument('--viz_freq', type=int, dest='viz_freq', default=20)
    # 
    parser.add_argument('--with_disc', action="store_true", dest='with_disc')
    args = parser.parse_args()

    WITH_DISC = args.with_disc
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

    net = ComposeNet()
    disc = Discriminator(3, args.img_size, 5, 143)

    initialize_model(net.feature_net.aux_convs)
    initialize_model(net.mask_net)
    initialize_model(net.edge_net)
    initialize_model(disc)

    net.cuda(args.gpu)
    disc.cuda(args.gpu)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    optim_disc = torch.optim.Adam(disc.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, 10, gamma=0.5)

    models = {
        "G": net, 
        "D": disc, 
    }

    optims = {
        "G": optim, 
        "D": optim_disc, 
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

