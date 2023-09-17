import math
import os
from datetime import datetime
from dataclasses import dataclass

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
from matplotlib import pyplot


@dataclass
class Config:
    dataset_path: str
    model_name: str
    model_version: str = 'v1'
    model_dir = "./models/"
    log_dir = "./logs/"
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_shape: tuple = 3, 224, 224
    latent_dim: int = 64

    epochs: int = 100
    batch_size: int = 32
    lr: float = 2e-4
    betas: tuple[float, ...] = None
    alphas: tuple[float, ...] = None
    lambdas: tuple[float, ...] = None

    mean: tuple[float, ...] = (0.5,) * image_shape[0]
    std: tuple[float, ...] = (0.5,) * image_shape[0]
    transforms: T.Compose = T.Compose([
        T.Resize(image_shape[1:][::-1]),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
    denormalize: T.Normalize = T.Normalize(-torch.tensor(mean) / torch.tensor(std), 1 / torch.tensor(std))

    @property
    def checkpoint_path(self):
        return f"{self.model_dir}/{self.model_name}/{self.model_version}/"

    @property
    def log_path(self):
        return f"{self.log_dir}/{self.model_name}/{self.model_version}/"

    def __post_init__(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.writer = SummaryWriter(self.log_path, filename_suffix=datetime.now().strftime('%Y%m%d-%H%M%S'))


@dataclass
class CycleGANConfig(Config):
    residuals: int = 9
    blocks: tuple = (64, 128, 256, 512)
    betas: tuple[float, float] = (0.5, 0.999)
    lambdas: tuple[float, float] = (10, 0.5)


def display_images(images):
    fig = pyplot.figure(figsize=(16, 16))
    rows = cols = int(math.ceil(len(images) ** 0.5))
    for i, image in enumerate(images):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(image)
    pyplot.show()


def save_checkpoint(
        path: str,
        models: dict[str, "nn.Module"],
        optimizers: dict[str, "optim.Optimizer"],
        **others,
) -> str:
    path = f"{path}/checkpoint-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pt" if not path.endswith(".pt") else path
    torch.save({
        **{f"model_{k}_state_dict": v.state_dict() for k, v in models.items()},
        **{f"optim_{k}_state_dict": v.state_dict() for k, v in optimizers.items()},
        "others": others if others is not None else {}
    }, path)
    return path


def load_checkpoint(
        path: str,
        models: dict[str, "nn.Module"],
        optimizers: dict[str, "optim.Optimizer"],
) -> dict:
    checkpoint = torch.load(path)
    for k, v in models.items():
        v.load_state_dict(checkpoint[f"model_{k}_state_dict"])
    for k, v in optimizers.items():
        v.load_state_dict(checkpoint[f"optim_{k}_state_dict"])
    return checkpoint["others"]


__all__ = [
    "Config",
    "CycleGANConfig",
    "display_images",
    "save_checkpoint",
    "load_checkpoint",
]
