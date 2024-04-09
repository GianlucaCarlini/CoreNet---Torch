import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from typing import Union, Any, Optional, Callable
import os


class ImageDataLoader(Dataset):
    """A PyTorch Dataset for loading images and masks.

    Parameters
    ----------
    images_path: str
        The path to the images.
    masks_path: str
        The path to the masks.
    geometric_transform: Callable
        The geometric transformations to apply to images and masks.
    color_transform: Callable
        The color transformations to apply to images.
    img_pre_processing: Callable
        The pre-processing to apply to the images.
    label_pre_processing: Callable
        The pre-processing to apply to the labels.
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        geometric_transform: Callable = None,
        color_transform: Callable = None,
        img_pre_processing: Callable = None,
        label_pre_processing: Callable = None,
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.geometric_transform = geometric_transform
        self.color_transform = color_transform
        self.img_pre_processing = img_pre_processing
        self.label_pre_processing = label_pre_processing

        self.ids = os.listdir(self.labels_dir)

        self.images = [os.path.join(self.images_dir, image_id) for image_id in self.ids]
        self.labels = [os.path.join(self.labels_dir, label_id) for label_id in self.ids]

        self.len = len(self.images)

    def __getitem__(self, index) -> tuple:
        image = PIL.Image.open(self.images[index]).convert("RGB")
        mask = PIL.Image.open(self.labels[index]).convert("L")

        if self.geometric_transform is not None:
            image, mask = self.geometric_transform(image, mask)

        if self.color_transform is not None:
            image = self.color_transform(image)

        if self.img_pre_processing is not None:
            image = self.img_pre_processing(image)

        if self.label_pre_processing is not None:
            mask = self.label_pre_processing(mask)

        return image, mask

    def __len__(self) -> int:
        return self.len
