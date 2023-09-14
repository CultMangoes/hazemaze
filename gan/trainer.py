from typing import TYPE_CHECKING, Callable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from .models import GANet

BAR_FORMAT = "{desc} {n_fmt}/{total_fmt}|{bar}|{percentage:3.0f}% [{elapsed}<{remaining}, {rate_fmt}{postfix}]"


def train(
        model: "GANet",
        ds: "Dataset",
        optimizer1: "optim.Optimizer",
        optimizer2: "optim.Optimizer",
        ne: int = 10, bs: int = 32,
        collate_fn=None
):
    dl = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate_fn)
    criterion = nn.BCELoss()
    # BCELoss: Y(i) * log(X(i)) + (1 - Y(i)) * log(1 - X(i))
    # loss = Ex->p(data) * log(D(X)) + Ex->p(z) * log(1 - D(G(z)))
    for epoch in range(ne):
        loss_sum = 0
        prog = tqdm(dl, desc=f"Epoch: 0/{ne} | Batch", postfix={"loss": "?"}, bar_format=BAR_FORMAT)
        for batch, DATA in enumerate(prog):  # type: ignore
            X = DATA["images"]
            N = torch.randn(X.shape[0], 128)

            Y = model(X, use="D")
            x = model(N, use="G")

            # discriminator
            optimizer1.zero_grad()
            y = model(x.detach(), use="D")
            loss = criterion(Y, torch.ones_like(Y)) + criterion(y, torch.zeros_like(y))
            loss /= 2
            loss.backward()
            optimizer1.step()
            loss_sum += loss.item()

            # generator
            optimizer2.zero_grad()
            y = model(x, use="D")
            loss = criterion(y, torch.ones_like(y))
            loss.backward()
            optimizer2.step()
            loss_sum += loss.item()

            prog.set_description(f"Epoch: {epoch + 1}/{ne} | Batch")
            prog.set_postfix(loss=f"{loss_sum / (batch + 1):.4f}")


__all__ = [
    "train"
]
