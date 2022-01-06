import os

import numpy as np
import torch
import torchvision.transforms.functional as TF

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_circle_param(n: int, min: int):
    half_n = n // 2
    radius = np.random.randint(low=min, high=half_n-min)
    center_x = radius + np.random.randint(low=0, high=n-2*radius)
    center_y = radius + np.random.randint(low=0, high=n-2*radius)
    return {
        "radius": radius,
        "x": center_x, 
        "y": center_y, 
    }

def generate_circle_img(n: int, x: int, y: int, radius: int, channel_size:int=3):
    sample = np.linspace(0, n-1, n)
    xv, yv = np.meshgrid(sample, sample)
    xv = xv - x
    yv = yv - y

    res = xv**2 + yv**2
    r_2 = radius**2

    circle_resion = res <= r_2
    background_resion = res > r_2

    res[circle_resion] = 255
    res[background_resion] = 0
    res = res.astype(np.uint8)
    # To 3 channels
    if channel_size == 3:
        res = np.stack([res, res, res], axis=-1)
    return res

def encode_circle_param(n: int, radius: torch.Tensor, center_x: torch.Tensor, center_y: torch.Tensor):
    half = n // 2
    c_radius = torch.log(radius / n)
    c_x = (center_x - half) / half
    c_y = (center_y - half) / half
    return {
        "radius": c_radius,
        "x": c_x, 
        "y": c_y, 
    }

def decode_circle_param(n: int, c_radius: torch.Tensor, c_center_x: torch.Tensor, c_center_y: torch.Tensor):
    half = n // 2
    radius = torch.exp(c_radius) * n
    center_x = c_center_x * half + half
    center_y = c_center_y * half + half
    return {
        "radius": radius,
        "x": center_x, 
        "y": center_y, 
    }

def generate_batch_circle(n: int, radius: torch.Tensor, center_x: torch.Tensor, center_y: torch.Tensor, channel_size:int=3):
    batch = []
    for r, x, y in zip(radius, center_x, center_y):
        batch.append(TF.to_tensor(generate_circle_img(n, x.item(), y.item(), r.item(), channel_size=channel_size)))
    batch = torch.stack(batch, dim=0)
    return batch
