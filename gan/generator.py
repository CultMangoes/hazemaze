from typing import Literal

from torch import nn


class ConvBlock(nn.Sequential):
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
            *ConvBlock(channels, channels, nn.ReLU(), **kwargs),
            *ConvBlock(channels, channels, **kwargs)
        )

    def forward(self, x):
        return super().forward(x) + x


class Generator(nn.Module):
    def __init__(self, inp_features: int, features: int, residuals: int, **kwargs):
        super().__init__()
        num_down = kwargs.pop("num_down", 2)
        num_up = kwargs.pop("num_up", 2)

        self.head = ConvBlock(inp_features, features, nn.ReLU(),
                              norm=2, kernel_size=7, stride=1, padding=3)
        self.down_blocks = nn.Sequential(
            *[ConvBlock(features * 2 ** i, features * 2 ** (i + 1), nn.ReLU(),
                        norm=2, kernel_size=3, stride=2, padding=1) for i in range(num_down)]
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(features * 4,
                            norm=2, kernel_size=3, stride=1, padding=1) for _ in range(residuals)]
        )
        self.up_blocks = nn.Sequential(
            *[ConvBlock(features * 2 ** (num_down - i), features * 2 ** (num_down - i - 1), nn.ReLU(),
                        norm=2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1) for i in range(num_up)]
        )
        self.tail = ConvBlock(features, inp_features, nn.Tanh(),
                              norm=0, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x = self.head(x)
        x = self.down_blocks(x)
        x = self.residual_blocks(x)
        x = self.up_blocks(x)
        x = self.tail(x)
        return x


def test_generator():
    import torch

    img_shape = 5, 224, 224
    gen = Generator(img_shape[0], 64, 9)
    print(gen)
    print(gen(torch.rand(1, *img_shape)).shape)
    print(gen(torch.rand(7, *img_shape)).shape)


if __name__ == '__main__':
    test_generator()
