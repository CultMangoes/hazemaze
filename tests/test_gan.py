import torch
import torchvision.transforms as T
from torch import optim

from gan import GANet, train
from datasets import CelebADataset

if __name__ == '__main__':
    img_shape = 3, 256, 256
    latent_dim = 128
    lr = 3e-4
    betas = 0.5, 0.999
    fixed_image = torch.rand(1, *img_shape)
    fixed_noise = torch.randn(1, latent_dim)

    img_transform = T.Compose([
        T.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    ds1 = CelebADataset("../../commons/datasets/CelebA", img_shape[1:], img_transform=img_transform)

    blocks = [img_shape[0], 16, 32, 64, 128, 256, 512]
    model = GANet(blocks, latent_dim=latent_dim)
    optimizer1 = optim.Adam(model.discriminator.parameters(), lr=lr, betas=betas)
    optimizer2 = optim.Adam(model.generator.parameters(), lr=lr, betas=betas)

    model.discriminator(fixed_image), model.generator(fixed_noise)

    train(
        model, ds1, optimizer1, optimizer2,
        ne=10, bs=32,
        collate_fn=ds1.collate_fn
    )
