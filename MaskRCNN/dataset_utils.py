import os

import data_utils

import numpy as np
import torch
import torch.utils.data
from PIL import Image

from data_utils import generate_random_data, masks_to_colorimg


class Segmentation_Dataset(torch.utils.data.Dataset):
    def __init__(self, height, width, num_imgs, transforms=None):
        self.height = height
        self.width = width
        self.num_imgs = num_imgs
        self.imgs, self.masks = generate_random_data(self.height, self.width, count=self.num_imgs)
        self.transforms = transforms

    def __getitem__(self, idx):
        # load images and masks
        img = self.imgs[idx]
        mask = self.masks[idx]
        # first id is the background, so remove it
        num_objs = mask.shape[1]
        boxes = []
        for i in range(num_objs):
            pos = np.where(mask[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        if self.transforms is not None:
            img = self.transforms(img)
            

        return img, target

    def __len__(self):
        return len(self.imgs)