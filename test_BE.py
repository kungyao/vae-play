import os
import argparse

import cv2
import torch
from torch._C import Value
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import DataLoader

from datasets.dataset import BEDataset
from models.networks_BE import ComposeNet
from tools.utils import makedirs

# Only return imgs and bimgs.
def test_collate_fn(batch):
    imgs, bimgs, eimgs = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    # bimgs = torch.stack(bimgs, dim=0)
    # eimgs = torch.stack(eimgs, dim=0)
    return imgs # , bimgs, eimgs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--path", type=str, dest='path', default="D:/Manga/bubble-gen-label", help="Data path")
    parser.add_argument("--model_path", type=str, dest='model_path', default=None, help="Model path")
    parser.add_argument('--img_size', type=int, dest='img_size', default=256)
    parser.add_argument('--gpu', type=int, dest='gpu', default=0)
    parser.add_argument('--batchsize', type=int, dest='batchsize', default=16)
    parser.add_argument('--debug', action="store_true", dest='debug')
    args = parser.parse_args()

    if args.debug:
        net = ComposeNet()
    else:
        if args.model_path is None:
            raise ValueError("args.model_path should not be None.")
        obj = torch.load(args.model_path, map_location=f"cuda:{args.gpu}")
        net = obj["networks"]
    res_output = "./results"
    makedirs(res_output)

    data_loader = DataLoader(
        BEDataset(args.path, (args.img_size, args.img_size), if_test=True), 
        batch_size=args.batchsize, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=test_collate_fn, 
        pin_memory=True)

    net.cuda(args.gpu)
    net.eval()
    with torch.no_grad():
        for i, (imgs) in enumerate(data_loader):
            b = imgs.size(0)
            imgs = imgs.cuda(args.gpu)

            preds = net(imgs)
            pred_edges = preds["edges"].cpu()
            pred_masks = preds["masks"].cpu()

            imgs = imgs.cpu()
            # To 3 channels.
            pred_edges = pred_edges.repeat(1, 3, 1, 1)
            pred_masks = pred_masks.repeat(1, 3, 1, 1)
            
            vutils.save_image(
                torch.cat([imgs, pred_masks, pred_edges], dim=0), 
                os.path.join(res_output, f"test_{i}.png"),
                nrow=b, 
                padding=2, 
                pad_value=1
            )
