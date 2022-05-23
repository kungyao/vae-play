import os
import argparse

import cv2
from numpy import dtype
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader

from datasets.dataset_font import KanaImageDataset
from models.networks_BE_font import ComposeNet
from tools.utils import makedirs

def save_test_batch(imgs, predictions, result_path, result_name):
    b = imgs.size(0)
    pred_edges = predictions["edges"].cpu()
    pred_masks = predictions["masks"].cpu()
    imgs = (imgs.cpu() * 255).to(dtype=torch.uint8)

    pred_edges = pred_edges.cpu()
    pred_masks = pred_masks.cpu()

    pred_edges[pred_edges>=0.5] = 1
    pred_edges[pred_edges<0.5] = 0

    pred_masks[pred_masks>=0.5] = 1
    pred_masks[pred_masks<0.5] = 0

    pred_edges = pred_edges.to(dtype=torch.bool)
    pred_masks = pred_masks.to(dtype=torch.bool)

    result_edges = []
    result_bubbles = []
    for j in range(b):
        result_edges.append(vutils.draw_segmentation_masks(imgs[j], masks=pred_edges[j], alpha=0.5, colors=(255, 0, 0)))
        result_bubbles.append(vutils.draw_segmentation_masks(imgs[j], masks=pred_masks[j], alpha=0.5, colors=(255, 0, 0)))
    result_edges = torch.stack(result_edges, dim=0).to(dtype=torch.float) / 255
    result_bubbles = torch.stack(result_bubbles, dim=0).to(dtype=torch.float) / 255

    # To 3 channels.
    # pred_edges = pred_edges.repeat(1, 3, 1, 1)
    pred_masks = pred_masks.repeat(1, 3, 1, 1)

    vutils.save_image(
        torch.cat([result_edges, result_bubbles, pred_masks], dim=0), 
        os.path.join(result_path, f"{result_name}.png"),
        nrow=b, 
        padding=2, 
        pad_value=1
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--path", type=str, dest='path', default="./crop_test_input", help="Data path")
    parser.add_argument("--model_path", type=str, dest='model_path', default=None, help="Model path")
    parser.add_argument('--img_size', type=int, dest='img_size', default=64)
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
        net = obj["networks"]["net"]
    res_output = "./results/BE_font"
    makedirs(res_output)

    transform = T.Compose([
        T.Resize((64, 64), interpolation=Image.NEAREST), 
        T.ToTensor(),
    ])

    data_loader = DataLoader(
        KanaImageDataset(args.path), 
        batch_size=args.batchsize, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=KanaImageDataset.collate_fn)

    net.cuda(args.gpu)
    net.eval()
    with torch.no_grad():
        for i, (imgs) in enumerate(data_loader):
            tensor_kana_imgs = []
            for _, kana_img in enumerate(imgs):
                tensor_kana_imgs.append(transform(kana_img))
            tensor_kana_imgs = torch.stack(tensor_kana_imgs, dim=0)
            tensor_kana_imgs = tensor_kana_imgs.cuda()

            preds = net(tensor_kana_imgs)
            save_test_batch(tensor_kana_imgs, preds, res_output, f"test_{i}")
            
