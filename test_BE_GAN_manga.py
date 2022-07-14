import argparse

import torch

from models.networks_BE_GAN import ComposeNet
from tools.utils import makedirs
from test_BE_manga import main_annotation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--path", type=str, dest='path', default="D:/KungYao/Manga/MangaDatabase190926", help="Data path")
    parser.add_argument("--anno_path", type=str, dest='anno_path', default="D:/KungYao/Manga/MangaDatabase190926-extra-test", help="Data path")
    parser.add_argument("--model_path", type=str, dest='model_path', default=None, help="Model path")
    parser.add_argument('--img_size', type=int, dest='img_size', default=512)
    parser.add_argument('--gpu', type=int, dest='gpu', default=0)
    args = parser.parse_args()

    if args.model_path is None:
        raise ValueError("args.model_path should not be None.")
    obj = torch.load(args.model_path, map_location=f"cuda:{args.gpu}")
    net = ComposeNet(3, args.img_size)
    net.load_state_dict(obj["networks"]["G"].state_dict())
    res_output = "./results/BE_GAN/manga"
    makedirs(res_output)

    # main_mask(args, net, res_output, None) "DragonBall", 
    # filter = ["AttackOnTitan", "InitialD", "KurokosBasketball", "OnePiece", "DragonBall"]
    filter = ["AttackOnTitan", "InitialD"]
    # filter = ["AttackOnTitan"]
    # filter = ["KurokosBasketball"]
    # filter = ["OnePiece"]
    # filter = ["DragonBall"]
    main_annotation(args, net, res_output, filter)
