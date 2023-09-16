from torch import nn


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight)
        if m.bias is not None: nn.init.constant_(m.bias, 0.0)
