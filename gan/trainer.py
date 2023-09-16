import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from tqdm import tqdm as TQDM

BAR_FORMAT = "{desc} {n_fmt}/{total_fmt}|{bar}|{percentage:3.0f}% [{elapsed}<{remaining}, {rate_fmt}{postfix}]"


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
            loss = trainer(DATA)
            loss_sum += loss.item()
            prog.set_description(f"Epoch: {epoch + 1}/{ne} | Batch")
            prog.set_postfix(loss=f"{loss_sum / (batch + 1):.4f}")


def get_cycle_gan_trainer(
        generator_A,
        generator_B,
        discriminator_A,
        discriminator_B,
        optimizer_G,
        optimizer_D,
        lambda_cycle: float = 10,
        lambda_identity: float = 0.5
):
    L1 = torch.nn.L1Loss()
    MSE = torch.nn.MSELoss()

    def trainer(DATA):
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

        # Backprop
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        loss_total.backward()
        optimizer_G.step()
        optimizer_D.step()

        return loss_total

    return trainer


__all__ = [
    "train",
    "get_cycle_gan_trainer"
]
