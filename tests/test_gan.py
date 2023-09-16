from torch import optim

from datasets import ITSDataset, DomainDataset
from gan.utils import Config
from gan.generator import Generator
from gan.discriminator import Discriminator
from gan.trainer import train, get_cycle_gan_trainer


config = Config(
    root="../../commons",
    model_name="HazeGan",
    dataset_name="its",
    epochs=1, batch_size=8,
    image_shape=(3, 64, 64),
)
config.residuals = 4
config.blocks = [64, 128, 256, 512]
config.cycle_lambda = 10
config.identity_lambda = 0.5

if __name__ == '__main__':
    ds = DomainDataset(
        ITSDataset(config.dataset_path, SET="hazy", download=True, img_transform=config.transforms, sub_sample=0.5),
        ITSDataset(config.dataset_path, SET="clear", download=True, img_transform=config.transforms, sub_sample=1)
    ).to(config.device)

    gen_A = Generator(config.image_shape[0], config.latent_dim, config.residuals)
    disc_A = Discriminator(config.image_shape[0], config.blocks)
    gen_B = Generator(config.image_shape[0], config.latent_dim, config.residuals)
    disc_B = Discriminator(config.image_shape[0], config.blocks)
    optimizer_G = optim.Adam(
        list(gen_A.parameters()) + list(gen_B.parameters()),
        lr=config.lr,
        betas=(config.beta_1, config.beta_2)
    )
    optimizer_D = optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()),
        lr=config.lr,
        betas=(config.beta_1, config.beta_2)
    )

    trainer = get_cycle_gan_trainer(gen_A, gen_B, disc_A, disc_B, optimizer_G, optimizer_D,
                                    lambda_cycle=config.cycle_lambda, lambda_identity=config.identity_lambda)

    train(
        trainer, ds,
        ne=config.epochs, bs=config.batch_size,
    )
