from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.transforms as T


@dataclass
class Config:
    root: str
    model_name: str
    dataset_name: str
    model_version: str = 'v1'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    image_shape: tuple = 3, 224, 224
    latent_dim: int = 64

    batch_size: int = 32
    epochs: int = 100
    lr: float = 0.0002
    beta_1: float = 0.5
    beta_2: float = 0.999

    transforms: T.Compose = T.Compose([
        T.Resize(image_shape[1:][::-1]),
        T.ToTensor(),
        T.Normalize((0.5,) * image_shape[0], (0.5,) * image_shape[0])
    ])

    @property
    def root_path(self):
        return Path(self.root)

    @property
    def dataset_path(self):
        return self.root_path / "datasets" / self.dataset_name

    @property
    def model_path(self):
        return self.root_path / "models" / self.model_name / self.model_version

    @property
    def log_path(self):
        return self.root_path / "logs" / self.model_name / self.model_version
