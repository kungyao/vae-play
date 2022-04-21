import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from skimage import measure

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

def find_contour(mask_img: np.ndarray):
    def select_contour(cs):
        if len(cs) == 1:
            return cs[0]
        select_i = -1
        max_area = 0
        for i, c in enumerate(cs):
            c = np.expand_dims(c.astype(np.float32), 1)
            c = cv2.UMat(c)
            area = abs(cv2.contourArea(c))
            if area > max_area:
                max_area = area
                select_i = i
        return cs[select_i]
    def process_contour(c):
        new_c = []
        for pt in c:
            y, x = pt
            y = round(y)
            x = round(x)
            new_item = [y, x]
            if len(new_c) != 0:
                if new_c[-1] == new_item:
                    continue
            new_c.append(new_item)
        # Remove end point because it is same as start point.
        del new_c[-1]
        return np.array(new_c)

    contour = measure.find_contours(mask_img, 0.8)
    if len(contour) != 0:
        contour = select_contour(contour)
        contour = process_contour(contour)
        if len(contour) != 0:
            # to [x, y]
            contour = np.array(np.flip(contour, axis=1))
    return contour

def resample_points(contour, max_points: int=256):
    l = len(contour)
    if l > max_points:
        sample_step = (l - 2) / (max_points - 2)
        select = np.arange(1, max_points - 1) * sample_step
        select = np.round(select, decimals=1)
        select = np.concatenate([[0], select, [l-1]], axis=0)
        # new_c = [contour[0]]
        # for i in range(max_points - 2):
        #     idx = round((i + 1) * sample_step)
        #     new_c.append(contour[idx])
        # new_c.append(contour[-1])
        return np.array(contour[select])
    return contour
    
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def rotate_vector(vx, vy, rotation):
    rvx = vx * np.cos(rotation) - vy * np.sin(rotation)
    rvy = vx * np.sin(rotation) + vy * np.cos(rotation)
    return rvx, rvy 
