import os
import sys
import time

import torch
import torchvision.transforms as T
from torchvision.utils import make_grid
from torchvision.io import read_video

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

load_checkpoint(
    "models/hazemaze/v1/checkpoint-2023-09-21 13_56_12.898988.pt",
    {"generatorA": generatorA, "generatorB": generatorB},
)


def image_dehazer(image: "Image", w=640):
    with torch.inference_mode():
        asp = image.size[1] / image.size[0]
        image = image.resize((w // 4 * 4, int(asp * w) // 4 * 4))
        image_arr = T.Normalize(config1.mean, config1.std)(T.ToTensor()(image).unsqueeze(0))
        t = time.time()
        dehazed_arr = generatorB(image_arr.to(config1.device)).cpu()
        t = time.time() - t
        grid_arr = make_grid(torch.cat([image_arr, dehazed_arr]), nrow=2, normalize=True)
        grid = T.ToPILImage()(grid_arr)
        return grid, t


def video_dehazer(video_file: str, w=360):
    with torch.inference_mode():
        video = read_video(video_file, start_pts=0, end_pts=10, pts_unit="sec")[0].permute(0, 3, 1, 2)
        video = video / video.max()
        asp = video.shape[2] / video.shape[3]
        w = min(w, video.shape[3])
        video = T.Resize((int(asp * w) // 4 * 4, w // 4 * 4))(video)
        video = T.Normalize(config1.mean, config1.std)(video)
        for image in video:
            dehazed_image = generatorB(image.unsqueeze(0).to(config1.device))
            grid_arr = make_grid(torch.stack([image, dehazed_image[0]]), nrow=2, normalize=True).cpu()
            grid = T.ToPILImage()(grid_arr)
            yield grid
