from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random
import time
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_SPLIT_SEED = 20260918
DEFAULT_RETENTIONS = (1.00, 0.75, 0.50, 0.25, 0.10)
DEFAULT_MIN_USER_TRAIN = 5


@dataclass(frozen=True)
class DatasetConfig:
    kind: str
    seeds: tuple[int, ...]
    batch_size: int
    max_epochs: int
    patience: int


DATASET_CONFIGS = {
    "100K": DatasetConfig("100K", (11, 22, 33, 44, 55, 66, 77, 88, 99, 110), 16384, 12, 3),
    "1M": DatasetConfig("1M", (11, 22, 33, 44, 55), 131072, 8, 2),
}


def _load_100k(data_dir: Path) -> pd.DataFrame:
    zip_path = data_dir / "ml-100k.zip"
    extracted = data_dir / "ml-100k" / "u.data"
    if not extracted.exists():
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing {zip_path}. See data/README.md")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
    return pd.read_csv(
        extracted,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )


def _load_1m(data_dir: Path) -> pd.DataFrame:
    ratings = data_dir / "ratings.dat"
    if not ratings.exists():
        zip_path = data_dir / "ml-1m.zip"
        extracted = data_dir / "ml-1m" / "ratings.dat"
        if extracted.exists():
            ratings = extracted
        elif zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(data_dir)
            ratings = extracted
        else:
            raise FileNotFoundError(f"Missing {ratings} or {zip_path}. See data/README.md")
    return pd.read_csv(
        ratings,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": np.int32, "item_id": np.int32, "rating": np.float32, "timestamp": np.int64},
    )


def load_dataset(kind: str, data_dir: Path):
    kind = kind.upper()
    if kind not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {kind}")
    df = _load_100k(data_dir) if kind == "100K" else _load_1m(data_dir)
    users = np.sort(df.user_id.unique())
    items = np.sort(df.item_id.unique())
    user_map = {u: i for i, u in enumerate(users)}
    item_map = {it: i for i, it in enumerate(items)}
    df["u"] = df.user_id.map(user_map).astype(np.int32)
    df["i"] = df.item_id.map(item_map).astype(np.int32)
    return df, len(users), len(items), DATASET_CONFIGS[kind]


def make_split(df: pd.DataFrame, split_seed: int = DEFAULT_SPLIT_SEED):
    train_idx, val_idx, test_idx = [], [], []
    for u, group in df.groupby("u", sort=False):
        idx = group.index.to_numpy()
        idx = np.random.default_rng(split_seed + int(u)).permutation(idx)
        n = len(idx)
        n_test = max(1, int(round(n * 0.10)))
        n_val = max(1, int(round(n * 0.10)))
        if n_test + n_val >= n:
            n_test, n_val = 1, 1
        test_idx.extend(idx[:n_test])
        val_idx.extend(idx[n_test : n_test + n_val])
        train_idx.extend(idx[n_test + n_val :])

    cols = ["u", "i", "rating"]
    train = df.loc[train_idx, cols].reset_index(drop=True)
    val = df.loc[val_idx, cols].reset_index(drop=True)
    test = df.loc[test_idx, cols].reset_index(drop=True)

    # Keep the validation/test protocol fixed to items observed in the full training pool.
    seen = set(train.i.unique())
    val = val[val.i.isin(seen)].reset_index(drop=True)
    test = test[test.i.isin(seen)].reset_index(drop=True)
    return train, val, test


def nested_orders(train: pd.DataFrame, split_seed: int = DEFAULT_SPLIT_SEED):
    orders = {}
    for u, group in train.groupby("u", sort=False):
        idx = group.index.to_numpy()
        orders[int(u)] = np.random.default_rng(split_seed + 500000 + int(u)).permutation(idx)
    return orders


def sparsify(train: pd.DataFrame, orders, frac: float, min_user_train: int = DEFAULT_MIN_USER_TRAIN):
    if np.isclose(frac, 1.0):
        return train.copy()
    keep = []
    for _, idx in orders.items():
        n_keep = max(min_user_train, int(round(len(idx) * frac)))
        keep.extend(idx[: min(n_keep, len(idx))])
    return train.loc[keep].reset_index(drop=True)


def to_tensors(df: pd.DataFrame, device: torch.device | str = "cpu"):
    return (
        torch.tensor(df.u.to_numpy(), dtype=torch.long, device=device),
        torch.tensor(df.i.to_numpy(), dtype=torch.long, device=device),
        torch.tensor(df.rating.to_numpy(), dtype=torch.float32, device=device),
    )


class BiasOnly(nn.Module):
    def __init__(self, n_users: int, n_items: int):
        super().__init__()
        self.ub = nn.Embedding(n_users, 1)
        self.ib = nn.Embedding(n_items, 1)
        nn.init.zeros_(self.ub.weight)
        nn.init.zeros_(self.ib.weight)

    def forward(self, u, i, mu):
        return mu + self.ub(u).squeeze(1) + self.ib(i).squeeze(1)


class BiasMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, k: int):
        super().__init__()
        self.user = nn.Embedding(n_users, k)
        self.item = nn.Embedding(n_items, k)
        self.ub = nn.Embedding(n_users, 1)
        self.ib = nn.Embedding(n_items, 1)
        nn.init.normal_(self.user.weight, std=0.05)
        nn.init.normal_(self.item.weight, std=0.05)
        nn.init.zeros_(self.ub.weight)
        nn.init.zeros_(self.ib.weight)

    def forward(self, u, i, mu):
        return mu + self.ub(u).squeeze(1) + self.ib(i).squeeze(1) + (self.user(u) * self.item(i)).sum(1)


class CosineMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, k: int):
        super().__init__()
        self.user = nn.Embedding(n_users, k)
        self.item = nn.Embedding(n_items, k)
        self.ub = nn.Embedding(n_users, 1)
        self.ib = nn.Embedding(n_items, 1)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(1.5), dtype=torch.float32))
        nn.init.normal_(self.user.weight, std=0.05)
        nn.init.normal_(self.item.weight, std=0.05)
        nn.init.zeros_(self.ub.weight)
        nn.init.zeros_(self.ib.weight)

    def forward(self, u, i, mu):
        pu = F.normalize(self.user(u), dim=1, eps=1e-8)
        qi = F.normalize(self.item(i), dim=1, eps=1e-8)
        alpha = F.softplus(self.log_alpha) + 1e-6
        return mu + self.ub(u).squeeze(1) + self.ib(i).squeeze(1) + alpha * (pu * qi).sum(1)


class NormalizedMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, k: int):
        super().__init__()
        self.user = nn.Embedding(n_users, k)
        self.item = nn.Embedding(n_items, k)
        self.ub = nn.Embedding(n_users, 1)
        self.ib = nn.Embedding(n_items, 1)
        self.log_scale = nn.Parameter(torch.zeros(k))
        nn.init.normal_(self.user.weight, std=0.05)
        nn.init.normal_(self.item.weight, std=0.05)
        nn.init.zeros_(self.ub.weight)
        nn.init.zeros_(self.ib.weight)

    def forward(self, u, i, mu):
        pu = F.normalize(self.user(u), dim=1, eps=1e-8)
        qi = F.normalize(self.item(i), dim=1, eps=1e-8)
        scale = F.softplus(self.log_scale) + 1e-6
        return mu + self.ub(u).squeeze(1) + self.ib(i).squeeze(1) + (pu * qi * scale).sum(1)


def make_model(name: str, n_users: int, n_items: int, k: int):
    if name == "BiasOnly":
        return BiasOnly(n_users, n_items)
    if name == "BiasMF":
        return BiasMF(n_users, n_items, k)
    if name == "CosineMF":
        return CosineMF(n_users, n_items, k)
    if name == "NormalizedMF":
        return NormalizedMF(n_users, n_items, k)
    raise ValueError(name)


@torch.no_grad()
def evaluate(model, tensors, mu: float, batch_size: int):
    u, i, y = tensors
    sum_sq = 0.0
    sum_abs = 0.0
    model.eval()
    for start in range(0, len(y), batch_size):
        end = min(start + batch_size, len(y))
        pred = model(u[start:end], i[start:end], mu)
        diff = pred - y[start:end]
        sum_sq += torch.sum(diff * diff).item()
        sum_abs += torch.sum(torch.abs(diff)).item()
    return math.sqrt(sum_sq / len(y)), sum_abs / len(y)


def fit_model(
    name: str,
    train_df: pd.DataFrame,
    val_tensors,
    test_tensors,
    n_users: int,
    n_items: int,
    seed: int,
    *,
    k: int = 32,
    lr: float = 0.015,
    weight_decay: float = 1e-5,
    batch_size: int,
    max_epochs: int,
    patience: int,
    device: torch.device | str = "cpu",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    train_tensors = to_tensors(train_df, device)
    mu = float(train_df.rating.mean())
    model = make_model(name, n_users, n_items, k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    u, i, y = train_tensors
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed * 1000 + epoch)).to(device)
        for start in range(0, len(y), batch_size):
            idx = order[start : start + batch_size]
            pred = model(u[idx], i[idx], mu)
            loss = ((pred - y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        val_rmse, _ = evaluate(model, val_tensors, mu, batch_size)
        if val_rmse < best - 1e-5:
            best = val_rmse
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("Training ended without a retained validation state")
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    test_rmse, test_mae = evaluate(model, test_tensors, mu, batch_size)
    return model, mu, {
        "rmse": test_rmse,
        "mae": test_mae,
        "best_epoch": best_epoch,
        "train_seconds": seconds,
        "val_rmse": best,
    }
