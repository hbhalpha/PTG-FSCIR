import torch
from PIL import Image
import numpy as np
import random
import math
import torchvision.transforms.functional as F


def add_masked_patch(masked_img, total_patch=256, masked_rate=0.75):
    patch_len = int(math.sqrt(total_patch))
    patch_width = 224 // patch_len
    masked_img = np.array(masked_img)

    masked_num = int(masked_rate * total_patch)
    patch_indices = np.random.choice(range(total_patch), masked_num, replace=False)

    # fill_color = (255, 255, 255)  # Use tuple for fill color

    result = torch.full((patch_width, patch_width, 3), fill_value=255, dtype=torch.uint8)
    fill_color = result.tolist()

    for idx in patch_indices:
        row = idx // patch_len
        col = idx % patch_len
        masked_img[row * patch_width: row * patch_width + patch_width,
        col * patch_width: col * patch_width + patch_width] = fill_color

    masked_img = Image.fromarray(masked_img)
    return masked_img


def add_masked_pixel(masked_img, masked_rate=0.75):
    num_elements = int(224 * 224 * masked_rate)
    indices = torch.randperm(224 * 224)[:num_elements]
    # fill_color = (255, 255, 255)  # Use tuple for fill color
    masked_img = np.array(masked_img)
    masked_img.view(-1, 3)[indices] = torch.tensor([255, 255, 255])
    masked_img.view(224, 224, -1)
    masked_img = Image.fromarray(masked_img)
    return masked_img


def add_atten_masked(masked_img, masked_list, total_patch=256):
    patch_len = int(math.sqrt(total_patch))
    patch_width = 224 // patch_len
    masked_img = np.array(masked_img)

    # masked_num = int(masked_rate * total_patch)
    # patch_indices = np.random.choice(range(total_patch), masked_num, replace=False)

    # fill_color = (255, 255, 255)  # Use tuple for fill color

    result = torch.full((patch_width, patch_width, 3), fill_value=255, dtype=torch.uint8)
    fill_color = result.tolist()
    for idx in masked_list:
        row = idx // patch_len
        col = idx % patch_len
        masked_img[row * patch_width: row * patch_width + patch_width,
        col * patch_width: col * patch_width + patch_width] = fill_color

    masked_img = Image.fromarray(masked_img)
    return masked_img

