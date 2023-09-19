import torch
from torch import optim
from torchvision import models
from torchvision.utils import make_grid, save_image

from __datasets__ import ITSDataset, DenseHazeCVPR2019Dataset, DomainDataset
from gan.utils import CycleGANConfig, save_checkpoint, load_checkpoint, display_images
from gan.generator import Generator
from gan.discriminator import Discriminator
from gan.trainer import PerceptualLoss, train, get_cycle_gan_trainer

config1 = CycleGANConfig(
    "../../commons/datasets/its/",
    "HazeGan",
    "v1",
    image_shape=(3, 64, 64),
    latent_dim=64,
    epochs=1, batch_size=8,
    lr=2e-4,
    betas=(0.5, 0.999),
    lambdas=(10, 0.5),
    residuals=7,
    blocks=(64, 128, 256, 512)
)
config2 = CycleGANConfig(
    "../../commons/datasets/dense_haze_cvpr2019/",
    "HazeGan",
    "v1",
    image_shape=(3, 512, 512),
    latent_dim=64,
    epochs=1, batch_size=8,
    lr=2e-4,
    betas=(0.5, 0.999),
    lambdas=(10, 0.5),
    residuals=7,
    blocks=(64, 128, 256, 512)
)

ds1 = DomainDataset(
    ITSDataset(config1.dataset_path, SET="hazy", download=True, img_transform=config1.transforms, sub_sample=0.2),
    ITSDataset(config1.dataset_path, SET="clear", download=True, img_transform=config1.transforms, sub_sample=1)
).to(config1.device)
ds2 = DomainDataset(
    DenseHazeCVPR2019Dataset(config2.dataset_path, SET="hazy", download=True, img_transform=config2.transforms, sub_sample=0.2),
    DenseHazeCVPR2019Dataset(config2.dataset_path, SET="GT", download=True, img_transform=config2.transforms, sub_sample=1)
).to(config1.device)

if __name__ == '__main__':
    pass
