import os
from pathlib import Path
from typing import Union, Literal

from gan.dataset import ImageDataset, DomainDataset


class CelebADataset(ImageDataset):
    @staticmethod
    def download(DIR: Union[str, "Path"]):
        user = input("Enter your kaggle username: ")
        token = input("Enter your kaggle token: ")
        os.environ["KAGGLE_USERNAME"] = user
        os.environ["KAGGLE_KEY"] = token

        import kaggle
        kaggle.api.dataset_download_files('jessicali9530/celeba-dataset', path=DIR, unzip=True, quiet=False)

        os.environ.pop("KAGGLE_USERNAME")
        os.environ.pop("KAGGLE_KEY")

    def get_data(self):
        files = os.listdir(self._DIR / "img_align_celeba" / "img_align_celeba")
        for file in files:
            yield self._DIR / "img_align_celeba" / "img_align_celeba" / file


class ITSDataset(ImageDataset):
    def __init__(
            self,
            DIR: Union[str, "Path"],
            img_transform=None,
            SET: Literal["hazy", "clear", "trans"] = "hazy",
            **kwargs
    ):
        assert SET in ("hazy", "clear", "trans"), \
            "invalid value of SET, must be one of 'haze', 'clear', 'trans'"
        self._set = SET
        super().__init__(DIR, img_transform=img_transform, **kwargs)

    @staticmethod
    def download(DIR: Union[str, "Path"]):
        user = input("Enter your kaggle username: ")
        token = input("Enter your kaggle token: ")
        os.environ["KAGGLE_USERNAME"] = user
        os.environ["KAGGLE_KEY"] = token

        import kaggle
        kaggle.api.dataset_download_files('balraj98/indoor-training-set-its-residestandard', path=DIR, unzip=True,
                                          quiet=False)

        os.environ.pop("KAGGLE_USERNAME")
        os.environ.pop("KAGGLE_KEY")

    def get_data(self):
        files = os.listdir(self._DIR / self._set)
        for file in files:
            yield self._DIR / self._set / file


__all__ = [
    "CelebADataset",
    "ITSDataset",
    "DomainDataset"
]
