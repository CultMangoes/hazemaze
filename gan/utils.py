import math
import os
from datetime import datetime
from dataclasses import dataclass

import torch
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

    batch_size: int = 32
    epochs: int = 100
    lr: float = 0.0002
    beta_1: float = 0.5
    beta_2: float = 0.999

    mean: tuple = (0.5,) * image_shape[0]
    std: tuple = (0.5,) * image_shape[0]
    transforms: T.Compose = T.Compose([
        T.Resize(image_shape[1:][::-1]),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
    denormalize: T.Normalize = T.Normalize(-torch.tensor(mean) / torch.tensor(std), 1 / torch.tensor(std))

    @property
    def model_path(self):
        return f"{self.model_dir}/{self.model_name}/{self.model_version}/"

    @property
    def log_path(self):
        return f"{self.log_dir}/{self.model_name}/{self.model_version}/"

    def __post_init__(self):
        os.makedirs(self.model_path, exist_ok=True)
        self.writer = SummaryWriter(self.log_path, comment=datetime.now().strftime('%Y%m%d-%H%M%S'))


def display_images(images):
    fig = pyplot.figure(figsize=(16, 16))
    rows = cols = int(math.ceil(len(images) ** 0.5))
    for i, image in enumerate(images):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(image)
    pyplot.show()


def save_checkpoint(models, optimizers, path):
    path = f"{path}/checkpoint-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pt" if not path.endswith(".pt") else path
    torch.save({
        'model_state_dict': [model.state_dict() for model in models],
        'optimizer_state_dict': [optimizer.state_dict() for optimizer in optimizers],
    }, path)
    return path


def load_checkpoint(models, optimizers, path):
    checkpoint = torch.load(path)
    for model, state_dict in zip(models, checkpoint['model_state_dict']):
        model.load_state_dict(state_dict)
    for optimizer, state_dict in zip(optimizers, checkpoint['optimizer_state_dict']):
        optimizer.load_state_dict(state_dict)


__all__ = [
    "Config",
    "display_images",
    "save_checkpoint",
    "load_checkpoint",
]
