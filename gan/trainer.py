import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from tqdm import tqdm as TQDM

BAR_FORMAT = "{desc} {n_fmt}/{total_fmt}|{bar}|{percentage:3.0f}% [{elapsed}<{remaining}, {rate_fmt}{postfix}]"


class PerceptualLoss:
    def __init__(self, model: "nn.Module", criterion=None):
        if criterion is None: criterion = nn.L1Loss()
        self.model = model.eval().requires_grad_(False)
        self.criterion = criterion

    def __call__(self, x, y):
        return self.criterion(self.model(x), self.model(y))


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
        generatorA: "nn.Module", generatorB: "nn.Module",
        discriminatorA: "nn.Module", discriminatorB: "nn.Module",
        optimizerG: "optim.Optimizer", optimizerD: "optim.Optimizer",
        perceptual_loss=None, lambda_cycle: float = 10, lambda_identity: float = 0.5,
        writer: "SummaryWriter" = None, period: int = 1,
        fixedA: "torch.Tensor" = None, fixedB: "torch.Tensor" = None,
):
    assert writer is None or (fixedA is not None and fixedB is not None), \
        "parameters `writer`, `fixedA` and `fixedB` are mutually inclusive"
    L1 = nn.L1Loss()
    MSE = nn.MSELoss()
    if writer is not None:
        grid_realA = make_grid(fixedA, nrow=1, normalize=True)
        grid_realB = make_grid(fixedB, nrow=1, normalize=True)
        # writer.add_graph(..., ...)

    def trainer(DATA, step):
        realA, realB = DATA["images_0"], DATA["images_1"]
        fakeA, fakeB = generatorA(realB), generatorB(realA)
        backA, backB = generatorA(fakeB), generatorB(fakeA)
        sameA, sameB = generatorA(realA), generatorB(realB)
        pred_realA, pred_realB = discriminatorA(realA), discriminatorB(realB)
        pred_fakeA_true, pred_fakeB_true = discriminatorA(fakeA.detach()), discriminatorB(fakeB.detach())
        pred_fakeA_false, pred_fakeB_false = discriminatorA(fakeA), discriminatorB(fakeB)

        # ===Discriminator Loss===
        # Adversarial Loss
        loss_adversarialDA = (MSE(pred_realA, torch.ones_like(pred_realA)) +
                              MSE(pred_fakeA_true, torch.zeros_like(pred_fakeA_true)))
        loss_adversarialDB = (MSE(pred_realB, torch.ones_like(pred_realB)) +
                              MSE(pred_fakeA_true, torch.zeros_like(pred_fakeA_true)))
        loss_adversarialD = (loss_adversarialDA + loss_adversarialDB) / 2
        # Total Loss
        lossD = loss_adversarialD
        # backprop
        optimizerD.zero_grad()
        lossD.backward()
        optimizerD.step()
        # ---End Discriminator Loss---

        # ===Generator Loss===
        # Adversarial Loss
        loss_adversarialGA = MSE(pred_fakeA_false, torch.ones_like(pred_fakeA_false))
        loss_adversarialGB = MSE(pred_fakeB_false, torch.ones_like(pred_fakeB_false))
        loss_adversarialG = (loss_adversarialGA + loss_adversarialGB) / 2
        # Cycle Loss
        loss_cycleA = L1(backA, realA)
        loss_cycleB = L1(backB, realB)
        loss_cycle = (loss_cycleA + loss_cycleB) / 2
        # Identity Loss
        loss_identityA = L1(sameA, realA)
        loss_identityB = L1(sameB, realB)
        loss_identity = (loss_identityA + loss_identityB) / 2
        # Perceptual Loss
        if perceptual_loss is not None:
            loss_perceptualA = perceptual_loss(fakeA, realA)
            loss_perceptualB = perceptual_loss(fakeB, realB)
            loss_perceptual = (loss_perceptualA + loss_perceptualB) / 2
        else:
            loss_perceptual = torch.tensor(0)
        # Total Loss
        lossG = (loss_adversarialG + lambda_cycle * loss_cycle + lambda_identity * loss_identity)
        # backprop
        optimizerG.zero_grad()
        lossG.backward()
        optimizerG.step()
        # ---End Generator Loss---

        loss_total = lossD + lossG
        if writer is not None and step % period == 0:
            writer.add_scalar("loss/adversarial_discriminator", loss_adversarialD.item(), step)
            writer.add_scalar("loss/adversarial_generator", loss_adversarialG.item(), step)
            writer.add_scalar("loss/cycle", loss_cycle.item(), step)
            writer.add_scalar("loss/identity", loss_identity.item(), step)
            writer.add_scalar("loss/perceptual", loss_perceptual.item(), step)
            writer.add_scalar("loss/total", loss_total.item(), step)
            with torch.inference_mode():
                grid_fakeA = make_grid(fakeA := generatorA(fixedB), nrow=1, normalize=True)
                grid_fakeB = make_grid(fakeB := generatorB(fixedA), nrow=1, normalize=True)
                grid_backA = make_grid(generatorA(fakeB), nrow=1, normalize=True)
                grid_backB = make_grid(generatorB(fakeA), nrow=1, normalize=True)
                grid_sameA = make_grid(generatorA(fixedA), nrow=1, normalize=True)
                grid_sameB = make_grid(generatorB(fixedB), nrow=1, normalize=True)
                writer.add_images("images/domainA",
                                  torch.stack([grid_realA, grid_fakeB, grid_backA, grid_sameA]), step)
                writer.add_images("images/domainB",
                                  torch.stack([grid_realB, grid_fakeA, grid_backB, grid_sameB]), step)

        return loss_total

    return trainer


__all__ = [
    "PerceptualLoss",
    "train",
    "get_cycle_gan_trainer"
]
