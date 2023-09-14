import os
from pathlib import Path
from typing import Union
from abc import ABCMeta, abstractmethod

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.io import read_image


class GANDataset(Dataset, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def get_data(DIR: Union[str, "Path"]) -> tuple[str]:
        raise NotImplementedError

    @staticmethod
    def download(DIR: Union[str, "Path"]):
        raise NotImplementedError("This dataset does not support download")

    @property
    def img_size(self):
        return self._img_size

    def __init__(
            self,
            DIR: Union[str, "Path"],
            img_size: tuple[int, int],
            img_transform=None,
            device=None,
            download: bool = False
    ):
        if os.path.isdir(DIR) and len(os.listdir(DIR)): download = False
        if download: self.download(DIR)
        assert os.path.isdir(DIR), \
            f"Directory {DIR} does not exist"
        assert len(img_size) == 2 and all(isinstance(i, int) for i in img_size), \
            f"Invalid img_size={img_size}, must be tuple of 2 ints"

        self._DIR = Path(DIR)
        self._img_size = img_size
        self._img_transform = img_transform
        self._device = device

        self._data = tuple(self.get_data(self._DIR))
        self._resizer = T.Resize(self._img_size[::-1], antialias=True)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item) -> dict[str, "torch.Tensor"]:
        try:
            item = iter(item)
            return self.collate_fn(self[idx] for idx in item)
        except TypeError:
            pass
        if isinstance(item, slice):
            return self.collate_fn(self[idx] for idx in range(*item.indices(len(self))))
        elif isinstance(item, int):
            file = self._data[item]
            img = read_image(str(file)) / 255
            if self._img_transform is not None: img = self._img_transform(img)
            img = self._resizer(img)
            return {
                "images": img.to(self._device)
            }
        else:
            raise TypeError(f"Invalid argument type {type(item)}")

    @staticmethod
    def collate_fn(batch) -> dict[str, "torch.Tensor"]:
        images, = list(zip(*(b.values() for b in batch)))
        images = torch.stack(images)
        return {
            "images": images,
        }

    def to(self, device):
        self._device = device
        return self


__all__ = [
    "GANDataset"
]
