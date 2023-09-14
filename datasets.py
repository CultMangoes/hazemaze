import os
from pathlib import Path
from typing import Union

from gan import GANDataset


class CelebADataset(GANDataset):
    @staticmethod
    def download(DIR: Union[str, "Path"]):
        user = input("Enter your kaggle username: ")
        token = input("Enter your kaggle token: ")
        os.environ["KAGGLE_USERNAME"] = user
        os.environ["KAGGLE_KEY"] = token
        os.makedirs(DIR, exist_ok=True)

        import kaggle
        kaggle.api.dataset_download_files('jessicali9530/celeba-dataset', path=DIR, unzip=True, quiet=False)

        os.environ.pop("KAGGLE_USERNAME")
        os.environ.pop("KAGGLE_KEY")

    @staticmethod
    def get_data(DIR: Union[str, "Path"]):
        DIR = Path(DIR)
        files = os.listdir(DIR / "img_align_celeba" / "img_align_celeba")
        for file in files:
            yield DIR / "img_align_celeba" / "img_align_celeba" / file
