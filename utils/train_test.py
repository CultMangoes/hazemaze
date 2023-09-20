from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from tqdm import tqdm as TQDM

BAR_FORMAT = "{desc} {n_fmt}/{total_fmt}|{bar}|{percentage:3.0f}% [{elapsed}<{remaining}, {rate_fmt}{postfix}]"


def train(
        trainer,
        ds: "Dataset",
        ne: int = 10, bs: int = 32,
        collate_fn=None,
        step_offset=0,
):
    dl = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate_fn)
    step = step_offset
    for epoch in range(ne):
        loss_sum = 0
        prog: "TQDM" = tqdm(dl, desc=f"Epoch: 0/{ne} | Batch", postfix={"loss": "?"}, bar_format=BAR_FORMAT)
        for batch, DATA in enumerate(prog):
            loss = trainer(DATA, step)
            loss_sum += loss.item()
            prog.set_description(f"Epoch: {epoch + 1}/{ne} | Batch")
            prog.set_postfix(loss=f"{loss_sum / (batch + 1):.4f}")
            step += 1
    return step


def test(
        tester,
        ds: "Dataset",
        bs: int = 32,
        collate_fn=None
):
    dl = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate_fn)
    loss_sum = 0
    prog: "TQDM" = tqdm(dl, desc=f"Batch", postfix={"loss": "?"}, bar_format=BAR_FORMAT)
    for batch, DATA in enumerate(prog):
        loss = tester(DATA)
        loss_sum += loss.item()
        prog.set_postfix(loss=f"{loss_sum / (batch + 1):.4f}")


__all__ = [
    "train",
    "test",
]
