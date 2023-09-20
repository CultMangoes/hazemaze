from typing import Literal

import torch
from torch import nn


class GConvBlock(nn.Sequential):
    NORM = [None, nn.BatchNorm2d, nn.InstanceNorm2d]

    def __init__(
            self,
            inp_channels: int, out_channels: int,
            activation: "nn.Module" = None,
            norm: Literal[0, 1, 2] = 0,
            **kwargs
    ):
        p = kwargs.pop("p", 0)
        n = kwargs.pop("n", 1)
        down = kwargs.pop("down", True)
        CONV = nn.Conv2d if down else nn.ConvTranspose2d

        layers = [
            CONV(inp_channels, out_channels, bias=not norm, padding_mode="reflect" if down else "zeros", **kwargs),
            self.NORM[norm](out_channels) if norm else None
        ]
        for _ in range(n - 1):
            layers.append(CONV(out_channels, out_channels, 3, 1, 1,
                               bias=not norm, padding_mode="reflect"))
            layers.append(self.NORM[norm](out_channels) if norm else None)
        layers.append(activation)
        layers.append(nn.Dropout(p) if p else None)

        super().__init__(*[layer for layer in layers if layer is not None])


class ResidualBlock(nn.Sequential):
    def __init__(self, channels: int, **kwargs):
        super().__init__(
            *GConvBlock(channels, channels, nn.ReLU(), **kwargs),
            *GConvBlock(channels, channels, **kwargs)
        )

    def forward(self, x):
        return super().forward(x) + x


class Generator(nn.Module):
    def __init__(self, inp_features: int, latent_dim: int, residuals: int, **kwargs):
        out_features = kwargs.pop("out_features", inp_features)
        n = kwargs.pop("n", 1)
        p = kwargs.pop("p", 0)
        norm = kwargs.pop("norm", 2)
        coder_len = kwargs.pop("coder_len", 2)
        super().__init__()

        self.head = GConvBlock(inp_features, latent_dim, nn.ReLU(),
                               norm=norm, kernel_size=7, stride=1, padding=3, n=n, p=p)
        self.down_blocks = nn.ModuleList([
            GConvBlock(latent_dim * 2 ** i, latent_dim * 2 ** (i + 1), nn.ReLU(),
                       norm=norm, kernel_size=3, stride=2, padding=1, n=n, p=p)
            for i in range(coder_len)
        ])
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(latent_dim * 4,
                            norm=norm, kernel_size=3, stride=1, padding=1, n=n, p=p)
              for _ in range(residuals)]
        )
        self.up_blocks = nn.ModuleList(reversed([
            GConvBlock(latent_dim * 2 ** (i + 1) * 2, latent_dim * 2 ** i, nn.ReLU(),
                       norm=norm, down=False, kernel_size=3, stride=2, padding=1, output_padding=1, n=n, p=p)
            for i in range(coder_len)
        ]))
        self.pred = GConvBlock(latent_dim, out_features, nn.Tanh(),
                               norm=0, kernel_size=7, stride=1, padding=3, n=n, p=p)

    def forward(self, x):
        x = self.head(x)
        skip_connections = []
        for down_block in self.down_blocks:
            x = down_block(x)
            skip_connections.append(x)
        x = self.residual_blocks(x)
        for skip_connection, up_block in zip(reversed(skip_connections), self.up_blocks):
            x = torch.cat([x, skip_connection], dim=1)
            x = up_block(x)
        return self.pred(x)


def test_generator():
    import torch

    img_shape = 5, 224, 224
    gen = Generator(img_shape[0], 64, 9)
    print(gen)
    print(gen(torch.rand(1, *img_shape)).shape)
    print(gen(torch.rand(7, *img_shape)).shape)


if __name__ == '__main__':
    test_generator()


__all__ = [
    "GConvBlock",
    "ResidualBlock",
    "Generator",
]
