import os
import random
from pathlib import Path
from typing import Union
from abc import ABCMeta, abstractmethod

import torch
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset, metaclass=ABCMeta):
    @abstractmethod
    def get_data(self) -> tuple[str]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def download(cls, DIR: Union[str, "Path"]):
        raise NotImplementedError("This dataset does not support download")

    @staticmethod
    def kaggle_download(DIR: Union[str, "Path"], dataset: str):
        user = input("Enter your kaggle username: ")
        token = input("Enter your kaggle token: ")
        os.environ["KAGGLE_USERNAME"] = user
        os.environ["KAGGLE_KEY"] = token
        import kaggle
        kaggle.api.dataset_download_files(dataset, path=DIR, unzip=True, quiet=False)
        os.environ.pop("KAGGLE_USERNAME")
        os.environ.pop("KAGGLE_KEY")

    def __init__(
            self,
            DIR: Union[str, "Path"],
            img_transform=None,
            **kwargs
    ):
        device = kwargs.pop("device", torch.device("cpu"))
        download = kwargs.pop("download", False)
        sub_sample = kwargs.pop("sub_sample", 1)
        if os.path.isdir(DIR) and len(os.listdir(DIR)): download = False
        if download:
            os.makedirs(DIR, exist_ok=True)
            self.download(DIR)
        assert os.path.isdir(DIR), \
            f"Directory {DIR} does not exist"
        assert 0 < sub_sample <= 1, \
            f"Value of sub_sample must be between 0 and 1, got {sub_sample}"

        self._DIR = Path(DIR)
        self._img_transform = img_transform
        self._device = device

        data = list(self.get_data())
        random.shuffle(data)
        self._data = data[:int(len(data) * sub_sample)]

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
            file = self._data[item % len(self)]
            img = Image.open(file)
            if self._img_transform is not None: img = self._img_transform(img)
            return {
                "images": img.to(self._device)
            }
        else:
            raise TypeError(f"Invalid argument type {type(item)}")

    @staticmethod
    def collate_fn(batch) -> dict[str, "torch.Tensor"]:
        return {
            "images": torch.stack([b["images"] for b in batch], dim=0)
        }

    def to(self, device):
        self._device = device
        return self


class DomainDataset(Dataset):
    def __init__(self, *domains: "ImageDataset", device=None):
        self._domains = domains
        self.to(device)

    def __len__(self):
        return max(len(d) for d in self._domains)

    def __getitem__(self, item):
        return {
            f"images_{i}": d[item]["images"] for i, d in enumerate(self._domains)
        }

    def to(self, device):
        for d in self._domains: d.to(device)
        return self


__all__ = [
    "ImageDataset",
    "DomainDataset"
]
