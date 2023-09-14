from typing import Union, Literal

from torch import nn


class CNNDiscriminator(nn.Module):
    def __init__(self, blocks):
        assert len(blocks) > 2
        super().__init__()
        self.layers = nn.Sequential(
            self._block(blocks[0], blocks[1], 4, 2, 1, nn.LeakyReLU(0.2), batch_norm=False),
            *[self._block(blocks[i + 1], n, 4, 2, 1, nn.LeakyReLU(0.2))
              for i, n in enumerate(blocks[2:])],
            self._block(blocks[-1], 1, 4, 2, 0, nn.Sigmoid(), batch_norm=False, n=1),
            nn.Flatten(-3, -1)
        )

    @staticmethod
    def _block(
            inp_channels: int,
            out_channels: int,
            kernel_size: Union[int, tuple[int, int]],
            stride: Union[int, tuple[int, int]],
            padding: Union[int, tuple[int, int]],
            activation: "nn.Module",
            batch_norm: bool = True,
            p: float = 0,
            n: int = 2
    ):
        layers = [
            nn.Conv2d(inp_channels, out_channels, kernel_size, stride, padding, bias=not batch_norm),
            *[nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=not batch_norm) for _ in range(n - 1)],
            nn.BatchNorm2d(out_channels) if batch_norm else None,
            activation,
            nn.Dropout(p) if p else None
        ]
        return nn.Sequential(*[layer for layer in layers if layer is not None])

    def forward(self, x):
        return self.layers(x)


class CNNGenerator(nn.Module):
    def __init__(self, latent_dim, blocks):
        assert len(blocks) > 2
        super().__init__()
        self.layers = nn.Sequential(
            nn.Unflatten(-1, (latent_dim, 1, 1)),
            self._block(latent_dim, blocks[0], 4, 1, 0, nn.ReLU(), batch_norm=False, n=1),
            *(self._block(blocks[i], n, 4, 2, 1, nn.ReLU())
              for i, n in enumerate(blocks[1:-1])),
            self._block(blocks[-2], blocks[-1], 4, 2, 1, nn.Tanh(), batch_norm=False)
        )

    @staticmethod
    def _block(
            inp_channels: int,
            out_channels: int,
            kernel_size: Union[int, tuple[int, int]],
            stride: Union[int, tuple[int, int]],
            padding: Union[int, tuple[int, int]],
            activation: "nn.Module",
            batch_norm: bool = True,
            p: float = 0,
            n: int = 2
    ):
        layers = [
            nn.ConvTranspose2d(inp_channels, out_channels, kernel_size, stride, padding, bias=not batch_norm),
            *[nn.ConvTranspose2d(out_channels, out_channels, 3, 1, 1, bias=not batch_norm) for _ in range(n - 1)],
            nn.BatchNorm2d(out_channels) if batch_norm else None,
            activation,
            nn.Dropout(p) if p else None
        ]
        return nn.Sequential(*[layer for layer in layers if layer is not None])

    def forward(self, x):
        return self.layers(x)


class GANet(nn.Module):
    def __init__(self, blocks, latent_dim: int = 128):
        super().__init__()
        self.discriminator = CNNDiscriminator(blocks)
        self.generator = CNNGenerator(latent_dim, blocks[::-1])

    def forward(self, x, use: Literal["G", "D"] = "G"):
        if use == "G":
            return self.generator(x)
        elif use == "D":
            return self.discriminator(x)
        else:
            raise ValueError(f"Invalid value for 'use': {use}")


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight)
        if m.bias is not None: nn.init.constant_(m.bias, 0.0)
