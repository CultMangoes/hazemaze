from dataclasses import dataclass

from torch import nn

from utils.config import Config


@dataclass
class CycleGANConfig(Config):
    latent_dim: int = 64
    coder_len: int = 2
    residuals: int = 9
    blocks: tuple = (64, 128, 256, 512)
    betas: tuple[float, float] = (0.5, 0.999)
    lambdas: tuple[float, float] = (10, 0.5)


def weights_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0.0)


__all__ = [
    "CycleGANConfig",
    "weights_init",
]
