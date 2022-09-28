import os
import argparse
from datetime import datetime

import torch
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from torch.nn.functional import grid_sample

from test_BE import save_test_batch
from datasets.dataset import BEGanDataset, ImageDataset
from models.networks_BE_GAN import *
from models.networks_BC import find_tensor_contour
from tools.ops import *
from tools.utils import makedirs

def train_collate_fn(batch):
    imgs, bimgs, eimgs, labels, contour_content, contour_boundary = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    bimgs = torch.stack(bimgs, dim=0)
    eimgs = torch.stack(eimgs, dim=0)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    return imgs, bimgs, eimgs, labels, contour_content, contour_boundary

def visual_data(imgs, bimgs, eimgs, labels, contour_content, contour_boundary, result_path="./", result_name="result"):
    import cv2
    b, c, h, w = imgs.shape
    
    result_contents = []
    result_boundarys = []
    for i in range(b):
        # 
        tmp_b = np.zeros((h, w, c), dtype=np.uint8)
        cnt_size = len(contour_content[i])
        cnt_x = (contour_content[i][:, 0] * 0.5 + 0.5) * h
        cnt_y = (contour_content[i][:, 1] * 0.5 + 0.5) * h
        for j in range(cnt_size):
            x_cur = int(cnt_x[j])
            y_cur = int(cnt_y[j])
            x_next = int(cnt_x[(j + 1)%cnt_size])
            y_next = int(cnt_y[(j + 1)%cnt_size])
            cv2.line(tmp_b, (x_cur, y_cur), (x_next, y_next), (255, 255, 255), thickness=1)
        tmp_b = TF.to_tensor(tmp_b)
        result_contents.append(tmp_b)
        # 
        tmp_b = np.zeros((h, w, c), dtype=np.uint8)
        cnt_size = len(contour_boundary[i])
        cnt_x = (contour_boundary[i][:, 0] * 0.5 + 0.5) * h
        cnt_y = (contour_boundary[i][:, 1] * 0.5 + 0.5) * h
        for j in range(cnt_size):
            x_cur = int(cnt_x[j])
            y_cur = int(cnt_y[j])
            x_next = int(cnt_x[(j + 1)%cnt_size])
            y_next = int(cnt_y[(j + 1)%cnt_size])
            cv2.line(tmp_b, (x_cur, y_cur), (x_next, y_next), (255, 255, 255), thickness=1)
        tmp_b = TF.to_tensor(tmp_b)
        result_boundarys.append(tmp_b)
    result_contents = torch.stack(result_contents, dim=0)
    result_boundarys = torch.stack(result_boundarys, dim=0)
    
    vutils.save_image(
        torch.cat([imgs, bimgs.repeat(1, 3, 1, 1), result_contents, eimgs.repeat(1, 3, 1, 1), result_boundarys], dim=0), 
        os.path.join(result_path, f"{result_name}.png"),
        nrow=b, 
        padding=2, 
        pad_value=1
    )
    return

def train(args, epoch, iterations, nets, optims, train_loader, aug_sets=None):
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
        "g_type_loss": 0, 
        "loss_cnt": 0, 
    }

    if aug_sets is not None:
        aug_dset, aug_iter, aug_dloader = aug_sets

    train_iter = iter(train_loader)
    for i in trange(iterations):
        if aug_sets is not None:
            if i % 10 == 0:
                try:
                    aug_imgs = next(aug_iter)
                except:
                    aug_iter = iter(aug_dloader)
                    aug_imgs = next(aug_iter)
                train_loader.dataset.synthesis_target = aug_imgs[0]

        try:
            imgs, bimgs, eimgs, labels, contour_content, contour_boundary = next(train_iter)
        except:
            train_iter = iter(train_loader)
            imgs, bimgs, eimgs, labels, contour_content, contour_boundary = next(train_iter)
        # visual_data(imgs, bimgs, eimgs, labels, contour_content, contour_boundary)
        # return
        b, c, h, w = imgs.shape
        # Prepare data
        imgs = imgs.cuda(args.gpu)
        # Bubble only mask
        bimgs = bimgs.cuda(args.gpu)
        # Boundary only image
        eimgs = eimgs.cuda(args.gpu)
        labels = labels.cuda(args.gpu)
        # contour_content = contour_content.cuda(args.gpu)
        # contour_boundary = contour_boundary.cuda(args.gpu)

        # D
        with torch.no_grad():
            preds = G(imgs)
            pred_masks = preds["masks"].sigmoid()
            pred_edges = preds["edges"].sigmoid()
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
        g_pred_type, g_pred_feats = D(imgs, pred_masks.sigmoid(), pred_edges.sigmoid())

        loss_mask = 0.5 * F.binary_cross_entropy_with_logits(pred_masks, bimgs) + compute_dice_loss(pred_masks.sigmoid(), bimgs)
        loss_egde = 0.5 * F.binary_cross_entropy_with_logits(pred_edges, eimgs) + compute_dice_loss(pred_edges.sigmoid(), eimgs)
        # 
        g_adv_loss = torch.mean(torch.abs(g_pred_feats - g_real_feats))
        g_type_loss = F.cross_entropy(g_pred_type, labels)
        # # 
        # pred_cnts = find_tensor_contour(pred_masks)
        # loss_cnt = 0
        # for idx in range(b):
        #     # 
        #     cnt_content = torch.stack([contour_content[idx][:, 0], contour_content[idx][:, 1]], dim=1)
        #     cnt_content = cnt_content.unsqueeze(0).unsqueeze(0).cuda(args.gpu)
        #     content_contour_sample = grid_sample(pred_masks[idx][None].sigmoid(), cnt_content, mode='bilinear')
        #     loss_cnt_c12 = F.l1_loss(content_contour_sample, torch.ones_like(content_contour_sample))
        #     # 
        #     normalized_pts = pred_cnts[idx]
        #     normalized_pts[:, 0] = (normalized_pts[:, 0] - w) / w
        #     normalized_pts[:, 1] = (normalized_pts[:, 1] - h) / h
        #     normalized_pts = normalized_pts.unsqueeze(0).unsqueeze(0).cuda(args.gpu)
        #     content_contour_sample = grid_sample(pred_masks[idx][None].sigmoid(), normalized_pts, mode='bilinear')
        #     content_contour_sample_target = grid_sample(bimgs[idx][None].sigmoid(), normalized_pts, mode='bilinear')
        #     loss_cnt_c21 = F.l1_loss(content_contour_sample, content_contour_sample_target)
        #     # 
        #     cnt_boundary = torch.stack([contour_boundary[idx][:, 0], contour_boundary[idx][:, 1]], dim=1)
        #     cnt_boundary = cnt_boundary.unsqueeze(0).unsqueeze(0).cuda(args.gpu)
        #     boundary_contour_sample = grid_sample(pred_edges[idx][None].sigmoid(), cnt_boundary, mode='bilinear')
        #     loss_cnt_b = F.l1_loss(boundary_contour_sample, torch.ones_like(boundary_contour_sample))
        #     loss_cnt = loss_cnt + loss_cnt_c12 + loss_cnt_c21 + loss_cnt_b
        # loss_cnt = loss_cnt / b
        loss_cnt = edge_loss(pred_masks.sigmoid(), bimgs) + edge_loss(pred_edges.sigmoid(), eimgs)
        losses = loss_mask * 2 + loss_egde * 2 + g_adv_loss + g_type_loss + loss_cnt * 0.5

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
        avg_loss["g_type_loss"] = (avg_loss["g_type_loss"] * count + g_type_loss.item()) / next_count
        avg_loss["loss_cnt"] = (avg_loss["loss_cnt"] * count + loss_cnt.item()) / next_count
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
    parser.add_argument('--aug_path', type=str, dest='aug_path', default=None)
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

    initialize_model(generator.aux_convs)
    initialize_model(generator.mask_net)
    initialize_model(generator.edge_net)
    initialize_model(discriminator)

    nets = {}
    nets["G"] = generator
    nets["D"] = discriminator

    optims = {}
    optims["G"] = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optims["D"] = torch.optim.Adam(discriminator.parameters(), lr=args.lr * 0.1, betas=(0.5, 0.999))

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

    aug_dset = None
    aug_iter = None
    aug_dloader = None
    if args.aug_path is not None:
        aug_dset = ImageDataset(args.aug_path)
        aug_dloader = DataLoader(
            aug_dset, 
            batch_size=1, 
            shuffle=True, 
            collate_fn=ImageDataset.collate_fn, 
            pin_memory=True)
        aug_iter = iter(aug_dloader)

    for epoch in range(args.epochs):
        if args.aug_path is not None:
            train(args, epoch, args.iterations, nets, optims, dloader, (aug_dset, aug_iter, aug_dloader))
        else:
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

