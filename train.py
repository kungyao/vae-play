import os
import argparse

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from tqdm import tqdm
from torch.utils.data import DataLoader

from tools.utils import makedirs, decode_circle_param, generate_batch_circle
from models.networks import VaeGan
from datasets.dataset import CDataset, CHANNEL_SIZE

def train(args, epoch, networks, optims, train_loader):
    lambda_mse = 1e-6

    VAE = networks["VAE"]

    OPTIM_ENC = optims["ENCODER"]
    OPTIM_DEC = optims["DECODER"]
    OPTIM_DISC = optims["DISCRIMINATOR"]
    OPTIM_AUX = optims["AUX"]

    count = 0
    avg_loss = {
        "loss_recon":0., 
        "loss_encoder":0., 
        "loss_discriminator":0., 
        "loss_decoder":0., 
        "loss_aux":0.
    }

    VAE.train()
    bar = tqdm(train_loader)
    bar.set_description(f"epoch[{epoch}];")
    # for i, (imgs, targets) in enumerate(train_loader):
    for i, (imgs, targets) in enumerate(bar):
        batch_size = imgs.size(0)

        imgs = imgs.cuda(args.gpu)
        targets = targets.cuda(args.gpu)

        x_tilde, disc_class, disc_layer, mus, log_variances, params = VAE(imgs)
              
        # split so we can get the different parts
        disc_layer_original = disc_layer[:batch_size]
        disc_layer_predicted = disc_layer[batch_size:-batch_size]
        disc_layer_sampled = disc_layer[-batch_size:]

        disc_class_original = disc_class[:batch_size]
        disc_class_predicted = disc_class[batch_size:-batch_size]
        disc_class_sampled = disc_class[-batch_size:]

        nle, kl, mse, bce_dis_original, bce_dis_predicted, bce_dis_sampled, l1_enc_param = VaeGan.loss(
            imgs, x_tilde, 
            disc_layer_original, disc_layer_predicted, disc_layer_sampled, 
            disc_class_original, disc_class_predicted, disc_class_sampled,
            mus, log_variances,
            targets, params)

        
        loss_recon = F.mse_loss(imgs, x_tilde)
        loss_encoder =  torch.sum(kl) + torch.sum(mse) # 
        loss_discriminator = torch.sum(bce_dis_original) + torch.sum(bce_dis_predicted) + torch.sum(bce_dis_sampled)
        loss_decoder = torch.sum(lambda_mse * mse) - (1.0 - lambda_mse) * loss_discriminator
        loss_aux = l1_enc_param
                                                                    
        VAE.zero_grad()
        loss_recon.backward(retain_graph=True)
        loss_encoder.backward(retain_graph=True) 
        loss_decoder.backward(retain_graph=True)
        loss_discriminator.backward(retain_graph=True)
        loss_aux.backward()

        OPTIM_ENC.step()
        OPTIM_DEC.step()
        OPTIM_DISC.step()
        OPTIM_AUX.step()
        
        next_count = count + batch_size
        avg_loss["loss_recon"] = (avg_loss["loss_recon"] * count + loss_recon.item()) / next_count
        avg_loss["loss_encoder"] = (avg_loss["loss_encoder"] * count + loss_encoder.item()) / next_count
        avg_loss["loss_decoder"] = (avg_loss["loss_decoder"] * count + loss_decoder.item()) / next_count
        avg_loss["loss_discriminator"] = (avg_loss["loss_discriminator"] * count + loss_discriminator.item()) / next_count
        avg_loss["loss_aux"] = (avg_loss["loss_aux"] * count + loss_aux.item()) / next_count
        count = next_count

        if (i+1) % args.viz_freq == 0:            
            print("")
            res_str = ""
            for key in avg_loss:
                res_str += f"{key}: {round(avg_loss[key], 6)}; "
            print(res_str)

            with torch.no_grad():
                x_tilde, _, _, _, _, params = VAE(imgs)
            rs, xs, ys = torch.unbind(params, dim=-1)
            params = decode_circle_param(args.img_size, rs, xs, ys)
            from_params = generate_batch_circle(args.img_size, params["radius"], params["x"], params["y"], channel_size=CHANNEL_SIZE)
            vutils.save_image(
                torch.cat([imgs.cpu(), x_tilde.cpu(), from_params], dim=0), 
                os.path.join(args.res_output, f"{epoch}_{i}.png"),
                nrow=batch_size, 
                padding=2, 
                pad_value=1
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    # 
    parser.add_argument('--epoch', type=int, dest='epochs', default=20)
    parser.add_argument('--batchsize', type=int, dest='batchsize', default=16)
    #
    parser.add_argument('--gpu', type=int, dest='gpu', default=0)
    # 
    parser.add_argument('--img_size', type=int, dest='img_size', default=128)
    parser.add_argument('--zdim', type=int, dest='zdim', default=128)
    #
    parser.add_argument('--res_output', type=str, dest='res_output', default='./results')
    parser.add_argument('--model_output', type=str, dest='model_output', default='./logs')
    parser.add_argument('--viz_freq', type=int, dest='viz_freq', default=16)
    args = parser.parse_args()

    makedirs(args.res_output)
    makedirs(args.model_output)

    vae_net = VaeGan(args.img_size, args.zdim, num_of_param=3)

    vae_net.cuda(args.gpu)
    networks = {
        "VAE": vae_net
    }

    # RMSprop
    lr = 1e-4
    optim_encoder = torch.optim.RMSprop(params=vae_net.encoder.parameters(), lr=lr)
    optim_decoder = torch.optim.RMSprop(params=vae_net.decoder.parameters(), lr=lr)
    optim_disc = torch.optim.RMSprop(params=vae_net.discriminator.parameters(), lr=lr)
    optim_aux = torch.optim.RMSprop(params=vae_net.param_encoder.parameters(), lr=lr)
    optims = {
        "ENCODER": optim_encoder, 
        "DECODER": optim_decoder, 
        "DISCRIMINATOR": optim_disc, 
        "AUX": optim_aux
    }

    # train_data = CDataset(128, min_radius=10, data_size=4096, ifGen=True, ifWrite=True)
    train_data = CDataset(128, ifGen=False, ifWrite=False)
    train_loader = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=4, pin_memory=True, collate_fn=CDataset.train_collate_fn)

    for epoch in range(args.epochs):
        train(args, epoch, networks, optims, train_loader)
        torch.save(
            {
                "networks": networks, 
                "optims": optims,
                "epoch": epoch
            }, 
            os.path.join(args.model_output, f"{epoch}.ckpt")
        )
