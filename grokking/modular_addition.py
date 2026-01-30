from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import TensorDataset


@dataclass(frozen=True)
class ModularAdditionSplit:
    p: int
    train_frac: float
    seed: int
    train: TensorDataset
    test: TensorDataset


def make_modular_addition_dataset(p: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return full modular addition dataset:
      inputs: LongTensor [p*p, 2] with pairs (a,b)
      labels: LongTensor [p*p] with (a+b) mod p
    """
    a = torch.arange(p, dtype=torch.long)
    b = torch.arange(p, dtype=torch.long)
    aa, bb = torch.meshgrid(a, b, indexing="ij")
    x = torch.stack([aa.reshape(-1), bb.reshape(-1)], dim=1)
    y = (x[:, 0] + x[:, 1]) % p
    return x, y


def make_split(*, p: int, train_frac: float, seed: int, device: torch.device) -> ModularAdditionSplit:
    if not (0.0 < train_frac < 1.0):
        raise ValueError(f"train_frac must be in (0,1), got {train_frac}")
    if p <= 1:
        raise ValueError(f"p must be > 1, got {p}")

    x, y = make_modular_addition_dataset(p)
    n = x.shape[0]
    n_train = int(round(train_frac * n))

    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    x_train, y_train = x[train_idx].to(device), y[train_idx].to(device)
    x_test, y_test = x[test_idx].to(device), y[test_idx].to(device)

    return ModularAdditionSplit(
        p=p,
        train_frac=train_frac,
        seed=seed,
        train=TensorDataset(x_train, y_train),
        test=TensorDataset(x_test, y_test),
    )


