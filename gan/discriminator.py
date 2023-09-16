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

        layers = [
            nn.Conv2d(inp_channels, out_channels, bias=not norm, padding_mode="reflect", **kwargs),
            self.NORM[norm](out_channels) if norm else None
        ]
        for _ in range(n - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1,
                                    bias=not norm, padding_mode="reflect"))
            layers.append(self.NORM[norm](out_channels) if norm else None)
        layers.append(activation)
        layers.append(nn.Dropout(p) if p else None)

        super().__init__(*[layer for layer in layers if layer is not None])


class Discriminator(nn.Module):
    @property
    def kwargs(self):
        return dict(norm=2, p=0, n=1, kernel_size=4, stride=2, padding=1)

    def __init__(self, inp_features: int, blocks: list[int]):
        super().__init__()
        assert len(blocks) > 2

        self.head = ConvBlock(inp_features, blocks[0], nn.LeakyReLU(0.2),
                              norm=0, kernel_size=4, stride=2, padding=1, p=0, n=1)
        self.blocks = nn.Sequential(*[
            ConvBlock(inp_features, out_features, nn.LeakyReLU(0.2), **self.kwargs)
            for inp_features, out_features in zip(blocks[:-2], blocks[1:-1])
        ])
        self.pred = nn.Sequential(
            *ConvBlock(blocks[-2], blocks[-1], nn.LeakyReLU(0.2),
                       norm=2, kernel_size=4, stride=1, padding=1, p=0, n=1),
            *ConvBlock(blocks[-1], 1, nn.Sigmoid(),
                       norm=0, kernel_size=4, stride=1, padding=1, p=0, n=1)
        )
        self.pred.__class__ = ConvBlock

    def forward(self, x):
        return self.pred(self.blocks(self.head(x)))


def test_discriminator():
    import torch

    img_shape = 5, 224, 224
    disc = Discriminator(img_shape[0], [64, 128, 256, 512])
    print(disc)
    print(disc(torch.rand(1, *img_shape)).shape)
    print(disc(torch.rand(7, *img_shape)).shape)


if __name__ == '__main__':
    test_discriminator()
