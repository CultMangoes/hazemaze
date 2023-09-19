import os
from pathlib import Path
from typing import Union, Literal

from gan.dataset import ImageDataset, DomainDataset


class CelebADataset(ImageDataset):
    @classmethod
    def download(cls, DIR: Union[str, "Path"]):
        cls.kaggle_download(DIR, "jessicali9530/celeba-dataset")

    def get_data(self):
        files = os.listdir(folder := self._DIR / "img_align_celeba" / "img_align_celeba")
        for file in files: yield folder / file


class ITSDataset(ImageDataset):
    SETS = "hazy", "clear", "trans"

    def __init__(
            self,
            DIR: Union[str, "Path"],
            img_transform=None,
            SET: Literal["hazy", "clear", "trans"] = "hazy",
            **kwargs
    ):
        assert SET in self.SETS, \
            f"invalid value of SET, must be one of {self.SETS}"
        self._set = SET
        super().__init__(DIR, img_transform=img_transform, **kwargs)

    @classmethod
    def download(cls, DIR: Union[str, "Path"]):
        cls.kaggle_download(DIR, "balraj98/indoor-training-set-its-residestandard")

    def get_data(self):
        files = os.listdir(folder := self._DIR / self._set)
        for file in files: yield folder / file


class DenseHazeCVPR2019Dataset(ImageDataset):
    SETS = "hazy", "GT"

    def __init__(
            self,
            DIR: Union[str, "Path"],
            img_transform=None,
            SET: Literal["hazy", "GT"] = "hazy",
            **kwargs
    ):
        assert SET in self.SETS, \
            f"invalid value of SET, must be one of {self.SETS}"
        self._set = SET
        super().__init__(DIR, img_transform=img_transform, **kwargs)

    @classmethod
    def download(cls, DIR: Union[str, "Path"]):
        cls.kaggle_download(DIR, "rajat95gupta/hazing-images-dataset-cvpr-2019")

    def get_data(self):
        files = os.listdir(folder := self._DIR / self._set)
        for file in files: yield folder / file


__all__ = [
    "CelebADataset",
    "ITSDataset",
    "DenseHazeCVPR2019Dataset",
    "DomainDataset"
]
