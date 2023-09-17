from torch import optim
from torchvision import models

from datasets import ITSDataset, DomainDataset
from gan.utils import CycleGANConfig, save_checkpoint, load_checkpoint
from gan.generator import Generator
from gan.discriminator import Discriminator
from gan.trainer import PerceptualLoss, train, get_cycle_gan_trainer

config = CycleGANConfig(
    "../../commons/datasets/its/",
    "HazeGan",
    "v1",
    image_shape=(3, 56, 56),
    latent_dim=64,
    epochs=1, batch_size=1,
    lr=2e-4,
    betas=(0.5, 0.999),
    lambdas=(10, 0.5),
    residuals=2,
    blocks=(64, 128, 256, 512)
)

if __name__ == '__main__':
    ds = DomainDataset(
        ITSDataset(config.dataset_path, SET="hazy", download=True, img_transform=config.transforms, sub_sample=0.25),
        ITSDataset(config.dataset_path, SET="clear", download=True, img_transform=config.transforms, sub_sample=1)
    ).to(config.device)

    gen_A = Generator(config.image_shape[0], config.latent_dim, config.residuals)
    gen_B = Generator(config.image_shape[0], config.latent_dim, config.residuals)
    disc_A = Discriminator(config.image_shape[0], list(config.blocks))
    disc_B = Discriminator(config.image_shape[0], list(config.blocks))
    optimizer_G = optim.Adam(
        list(gen_A.parameters()) + list(gen_B.parameters()),
        lr=config.lr,
        betas=config.betas
    )
    optimizer_D = optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()),
        lr=config.lr,
        betas=config.betas
    )

    perceptual_model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:35].to(config.device)
    perceptual_loss = PerceptualLoss(perceptual_model)
    fixed_A, fixed_B = ds[:4].values()
    trainer = get_cycle_gan_trainer(gen_A, gen_B, disc_A, disc_B, optimizer_G, optimizer_D,
                                    lambda_cycle=config.lambdas[0], lambda_identity=config.lambdas[1],
                                    perceptual_loss=perceptual_loss, writer=config.writer,
                                    fixed_A=fixed_A, fixed_B=fixed_B, period=1)

    train(
        trainer, ds,
        ne=config.epochs, bs=config.batch_size,
    )

    # file_path = save_checkpoint(
    #     [gen_A, gen_B, disc_A, disc_B],
    #     [optimizer_G, optimizer_D],
    #     config.model_path
    # )
    # load_checkpoint(
    #     [gen_A, gen_B, disc_A, disc_B],
    #     [optimizer_G, optimizer_D],
    #     file_path,
    # )
