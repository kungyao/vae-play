import os
import argparse
from datetime import datetime

import torch
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from tqdm import tqdm, trange
from torch.utils.data import DataLoader

from datasets .dataset import BEDatasetGAN
from models.network_Style_GAN import *
from tools.ops import *
from tools.utils import makedirs

def train_collate_fn(batch):
    imgs, bimgs, eimgs, labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    bimgs = torch.stack(bimgs, dim=0)
    eimgs = torch.stack(eimgs, dim=0)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    return imgs, bimgs, eimgs, labels

def save_test_batch(x_org, x_ref, x_rec, x_stl, result_path, result_name):
    b = x_org.size(0)

    vutils.save_image(
        torch.cat([x_org, x_ref, x_rec, x_stl], dim=0), 
        os.path.join(result_path, f"{result_name}.png"),
        nrow=b, 
        padding=2, 
        pad_value=1
    )

def train(args, epoch, iterations, nets, optims, train_loader):
    G = nets["G"]
    E = nets["E"]
    D = nets["D"]

    g_opt = optims["G"]
    e_opt = optims["E"]
    d_opt = optims["D"]

    G.train()
    E.train()
    D.train()
    
    count = 0
    avg_loss = {
        "loss_d_adv": 0,
        "loss_g_adv": 0, 
        "loss_g_rec": 0, 
        "loss_c_stl": 0, 
    }

    train_iter = iter(train_loader)
    for i in trange(iterations):
        try:
            imgs, bimgs, eimgs, labels = next(train_iter)
        except:
            train_iter = iter(train_loader)
            imgs, bimgs, eimgs, labels = next(train_iter)

        # b = imgs.size(0)
        # imgs = imgs.cuda(args.gpu)
        # bimgs = bimgs.cuda(args.gpu)
        # eimgs = eimgs.cuda(args.gpu)
        # labels = labels.cuda(args.gpu)

        # Prepare data
        x_org = imgs.cuda(args.gpu)
        y_org = labels.cuda(args.gpu)

        # Prepare style data
        x_ref_idx = torch.randperm(x_org.size(0))
        x_ref_idx = x_ref_idx.cuda(args.gpu)
        x_ref = x_org.clone()
        x_ref = x_ref[x_ref_idx]
        y_ref = y_org.clone()
        y_ref = y_ref[x_ref_idx]

        ###
        # D
        ###
        with torch.no_grad():
            s_ref = E(x_ref)
            x_fake = G(x_org, s_ref)
        
        x_ref.requires_grad_()
        d_real_logit = D(x_ref, y_ref)
        d_fake_logit = D(x_fake, y_ref)

        d_adv_real = compute_hinge_loss(d_real_logit, 'd_real')
        d_adv_fake = compute_hinge_loss(d_fake_logit, 'd_fake')
        d_adv_loss = d_adv_real + d_adv_fake

        d_opt.zero_grad()
        d_adv_loss.backward()
        d_opt.step()

        ###
        # G
        ###
        s_org = E(x_org)
        s_ref = E(x_ref)

        d1_org, d2_org, d3_org, d4_org, d5_org = G.encode(x_org)
        # d1_stl, d2_stl, d3_stl, d4_stl, d5_stl = G.encode(x_stl)
        x_rec = G.decode(d1_org, d2_org, d3_org, d4_org, d5_org, s_org)
        x_stl = G.decode(d1_org, d2_org, d3_org, d4_org, d5_org, s_ref)

        g_rec_logit = D(x_rec, y_org)
        g_stl_logit = D(x_stl, y_ref)

        g_adv_rec = compute_hinge_loss(g_rec_logit, 'g')
        g_adv_stl = compute_hinge_loss(g_stl_logit, 'g')
        g_adv_loss = g_adv_rec + g_adv_stl

        g_rec_loss = F.mse_loss(x_rec, x_org)

        _, _, _, _, d5_stl = G.encode(x_stl)
        g_c_stl_loss = F.mse_loss(d5_stl, d5_org)

        g_loss = g_adv_loss + g_rec_loss * 0.1 + g_c_stl_loss * 0.1

        g_opt.zero_grad()
        e_opt.zero_grad()
        g_loss.backward()
        g_opt.step()
        e_opt.step()
        
        # 
        next_count = count + imgs.size(0)
        avg_loss["loss_d_adv"] = (avg_loss["loss_d_adv"] * count + d_adv_loss.item()) / next_count
        avg_loss["loss_g_adv"] = (avg_loss["loss_g_adv"] * count + g_adv_loss.item()) / next_count
        avg_loss["loss_g_rec"] = (avg_loss["loss_g_rec"] * count + g_rec_loss.item()) / next_count
        avg_loss["loss_c_stl"] = (avg_loss["loss_c_stl"] * count + g_c_stl_loss.item()) / next_count
        count = next_count

        if (i+1) % args.viz_freq == 0:
            print("")
            res_str = ""
            for key in avg_loss:
                res_str += f"{key}: {round(avg_loss[key], 6)}; "
            print(res_str)
            save_test_batch(x_org, x_ref, x_rec, x_stl, args.res_output, f"{epoch}_{i+1}")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path', type=str, dest='path', default="D:/Manga/bubble-gen-label")
    # 
    parser.add_argument('--lr', type=float, dest='lr', default=1e-4)
    parser.add_argument('--gpu', type=int, dest='gpu', default=0)
    parser.add_argument('--epochs', type=int, dest='epochs', default=2)
    parser.add_argument('--iterations', type=int, dest='iterations', default=1000)
    parser.add_argument('--batchsize', type=int, dest='batchsize', default=32)
    #
    parser.add_argument('--workers', type=int, dest='workers', default=0)
    # 
    parser.add_argument('--z_dim', type=int, dest='z_dim', default=512)
    parser.add_argument('--img_size', type=int, dest='img_size', default=256)
    # parser.add_argument('--max_points', type=int, dest='max_points', default=DEFAULT_MAX_POINTS)
    #
    parser.add_argument('--res_output', type=str, dest='res_output', default='./results')
    parser.add_argument('--model_output', type=str, dest='model_output', default='./logs')
    parser.add_argument('--viz_freq', type=int, dest='viz_freq', default=100)
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
    
    generator = Generator(args.z_dim)
    style_encoder = StyleEncoder(args.z_dim, args.img_size)
    discriminator = Discriminator(args.img_size, 3)

    initialize_model(generator)
    initialize_model(style_encoder)
    initialize_model(discriminator)

    generator.cuda(args.gpu)
    style_encoder.cuda(args.gpu)
    discriminator.cuda(args.gpu)

    nets = {}
    nets["G"] = generator
    nets["E"] = style_encoder
    nets["D"] = discriminator

    optims = {}
    optims["G"] = torch.optim.RMSprop(generator.parameters(), lr=args.lr, weight_decay=0.0001)
    optims["E"] = torch.optim.Adam(style_encoder.parameters(), lr=args.lr)
    optims["D"] = torch.optim.RMSprop(discriminator.parameters(), lr=args.lr, weight_decay=0.0001)

    for name, net in nets.items():
        nets[name] = net.cuda(args.gpu)

    dset = BEDatasetGAN(args.path, (args.img_size, args.img_size))
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

