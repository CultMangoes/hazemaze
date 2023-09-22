import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as T
from torchvision.utils import make_grid

from gan import CycleGANConfig, Generator
from utils.checkpoints import load_checkpoint
from PIL import Image

config1 = CycleGANConfig(
    "none",
    "HazeGan",
    "v1",
    image_shape=(3, 64, 64),
    latent_dim=64,
    dropout=0,
    num_epochs=1, batch_size=8,
    lr=2e-4,
    betas=(0.5, 0.999),
    lambdas=(10, 0.5),
    residuals=5,
    blocks=(64, 128, 256, 512),
    writer=False,
)

generatorA = Generator(
    config1.image_shape[0],
    config1.latent_dim,
    config1.residuals,
    p=config1.dropout,
    coder_len=config1.coder_len,
).to(config1.device).eval()

generatorB = Generator(
    config1.image_shape[0],
    config1.latent_dim,
    config1.residuals,
    p=config1.dropout,
    coder_len=config1.coder_len,
).to(config1.device).eval()

others = load_checkpoint(
    "models/hazemaze/v1/checkpoint-2023-09-21 13_56_12.898988.pt",
    {"generatorA": generatorA, "generatorB": generatorB},
)


def dehazer(image: "Image"):
    with torch.inference_mode():
        w = 640
        asp = image.size[1] / image.size[0]
        image = image.resize((w // 4 * 4, int(asp * w) // 4 * 4))
        image_arr = T.Normalize(config1.mean, config1.std)(T.ToTensor()(image).unsqueeze(0))
        t = time.time()
        dehazed_arr = generatorB(image_arr.to(config1.device)).cpu()
        t = time.time() - t
        grid_arr = make_grid(torch.cat([image_arr, dehazed_arr]), nrow=2, normalize=True)
        grid = T.ToPILImage()(grid_arr)
        return grid, t
