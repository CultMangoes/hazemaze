import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from tqdm import tqdm as TQDM

BAR_FORMAT = "{desc} {n_fmt}/{total_fmt}|{bar}|{percentage:3.0f}% [{elapsed}<{remaining}, {rate_fmt}{postfix}]"


class PerceptualLoss:
    def __init__(self, model: "nn.Module", loss_fn=None):
        if loss_fn is None: loss_fn = nn.L1Loss()
        self.model = model.eval().requires_grad_(False)
        self.loss = loss_fn

    def __call__(self, x, y):
        return self.loss(self.model(x), self.model(y))


def train(
        trainer,
        ds: "Dataset",
        ne: int = 10, bs: int = 32,
        collate_fn=None
):
    dl = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate_fn)
    for epoch in range(ne):
        loss_sum = 0
        prog: "TQDM" = tqdm(dl, desc=f"Epoch: 0/{ne} | Batch", postfix={"loss": "?"}, bar_format=BAR_FORMAT)
        for batch, DATA in enumerate(prog):
            loss = trainer(DATA, epoch * len(dl) + batch)
            loss_sum += loss.item()
            prog.set_description(f"Epoch: {epoch + 1}/{ne} | Batch")
            prog.set_postfix(loss=f"{loss_sum / (batch + 1):.4f}")


def get_cycle_gan_trainer(
        generator_A: "nn.Module", generator_B: "nn.Module",
        discriminator_A: "nn.Module", discriminator_B: "nn.Module",
        optimizer_G: "optim.Optimizer", optimizer_D: "optim.Optimizer",
        perceptual_loss=None, lambda_cycle: float = 10, lambda_identity: float = 0.5,
        writer: "SummaryWriter" = None, period: int = 1,
        fixed_A: "torch.Tensor" = None, fixed_B: "torch.Tensor" = None,
):
    assert writer is None or (fixed_A is not None and fixed_B is not None), \
        "parameters `writer`, `fixed_A` and `fixed_B` are mutually inclusive"
    L1 = nn.L1Loss()
    MSE = nn.MSELoss()
    if writer is not None:
        grid_real_A = make_grid(fixed_A, nrow=1, normalize=True)
        grid_real_B = make_grid(fixed_B, nrow=1, normalize=True)
        # writer.add_graph(..., ...)

    def trainer(DATA, step):
        real_A, real_B = DATA["images_0"], DATA["images_1"]
        fake_A, fake_B = generator_A(real_B), generator_B(real_A)
        recon_A, recon_B = generator_A(fake_B), generator_B(fake_A)
        same_A, same_B = generator_A(real_A), generator_B(real_B)
        pred_real_A, pred_real_B = discriminator_A(real_A), discriminator_B(real_B)
        pred_fake_A_true, pred_fake_B_true = discriminator_A(fake_A.detach()), discriminator_B(fake_B.detach())
        pred_fake_A_false, pred_fake_B_false = discriminator_A(fake_A), discriminator_B(fake_B)

        # Discriminator Loss
        loss_D_A = (MSE(pred_real_A, torch.ones_like(pred_real_A)) +
                    MSE(pred_fake_A_true, torch.zeros_like(pred_fake_A_true)))
        loss_D_B = (MSE(pred_real_B, torch.ones_like(pred_real_B)) +
                    MSE(pred_fake_A_true, torch.zeros_like(pred_fake_A_true)))
        loss_D = (loss_D_A + loss_D_B) / 2

        # Generator Loss
        loss_G_A = MSE(pred_fake_A_false, torch.ones_like(pred_fake_A_false))
        loss_G_B = MSE(pred_fake_B_false, torch.ones_like(pred_fake_B_false))
        loss_G = (loss_G_A + loss_G_B) / 2

        # Cycle Loss
        loss_cycle_A = L1(recon_A, real_A)
        loss_cycle_B = L1(recon_B, real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Identity Loss
        loss_identity_A = L1(same_A, real_A)
        loss_identity_B = L1(same_B, real_B)
        loss_identity = (loss_identity_A + loss_identity_B) / 2

        # Total Loss
        loss_total = loss_D + loss_G + lambda_cycle * loss_cycle + lambda_identity * loss_identity

        if perceptual_loss is not None:
            loss_perceptual_A = perceptual_loss(fake_A, real_A)
            loss_perceptual_B = perceptual_loss(fake_B, real_B)
            loss_perceptual = (loss_perceptual_A + loss_perceptual_B) / 2
            loss_total += loss_perceptual
        else:
            loss_perceptual = torch.tensor(torch.nan)

        # Backprop
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        loss_total.backward()
        optimizer_G.step()
        optimizer_D.step()

        if writer is not None and step % period == 0:
            writer.add_scalar("loss/discriminator", loss_D.item(), step)
            writer.add_scalar("loss/generator", loss_G.item(), step)
            writer.add_scalar("loss/cycle", loss_cycle.item(), step)
            writer.add_scalar("loss/identity", loss_identity.item(), step)
            writer.add_scalar("loss/perceptual", loss_perceptual.item(), step)
            writer.add_scalar("loss/total", loss_total.item(), step)

            with torch.inference_mode():
                grid_fake_A = make_grid(fake_A := generator_A(fixed_B), nrow=1, normalize=True)
                grid_fake_B = make_grid(fake_B := generator_B(fixed_A), nrow=1, normalize=True)
                grid_recon_A = make_grid(generator_A(fake_B), nrow=1, normalize=True)
                grid_recon_B = make_grid(generator_B(fake_A), nrow=1, normalize=True)
                grid_same_A = make_grid(generator_A(fixed_A), nrow=1, normalize=True)
                grid_same_B = make_grid(generator_B(fixed_B), nrow=1, normalize=True)
                writer.add_images("images/domainA",
                                  torch.stack([grid_real_A, grid_fake_B, grid_recon_A, grid_same_A]), step)
                writer.add_images("images/domainB",
                                  torch.stack([grid_real_B, grid_fake_A, grid_recon_B, grid_same_B]), step)

        return loss_total

    return trainer


__all__ = [
    "PerceptualLoss",
    "train",
    "get_cycle_gan_trainer"
]
