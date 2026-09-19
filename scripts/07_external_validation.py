from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lfn.core import (
    DEFAULT_RETENTIONS,
    DEFAULT_SPLIT_SEED,
    fit_model,
    make_split,
    nested_orders,
    sparsify,
    to_tensors,
)

MODELS = ("BiasOnly", "BiasMF", "CosineMF", "NormalizedMF")
SEEDS_5 = (11, 22, 33, 44, 55)
SEEDS_3 = (11, 22, 33)
K = 32
LR = 0.015
WEIGHT_DECAY = 1e-5
MIN_USER_TRAIN = 5


def reindex(df: pd.DataFrame):
    users = np.sort(df.user_id.unique())
    items = np.sort(df.item_id.unique())
    um = {u: i for i, u in enumerate(users)}
    im = {it: i for i, it in enumerate(items)}
    out = df.copy()
    out["u"] = out.user_id.map(um).astype(np.int32)
    out["i"] = out.item_id.map(im).astype(np.int32)
    out["rating"] = out.rating.astype(np.float32)
    return out, len(users), len(items)


def scale_linear(x: pd.Series, lo: float, hi: float) -> pd.Series:
    return 1.0 + (x.astype(float) - lo) * 4.0 / (hi - lo)


def load_book_crossing(data_dir: Path):
    p = data_dir / "BX-Book-Explicit-5Rate-Map.csv"
    x = pd.read_csv(p).rename(columns={"uid": "user_id", "bid": "item_id"})
    # >=7 observations permits at least 5 train + 1 validation + 1 test.
    counts = x.groupby("user_id").size()
    x = x[x.user_id.isin(counts[counts >= 7].index)].copy()
    x["rating"] = scale_linear(x["rating"], 1.0, 10.0)
    return reindex(x)


def load_filmtrust(data_dir: Path):
    p = data_dir / "ratings.txt"
    x = pd.read_csv(p, sep=r"\s+", header=None, names=["user_id", "item_id", "rating"])
    counts = x.groupby("user_id").size()
    x = x[x.user_id.isin(counts[counts >= 7].index)].copy()
    x["rating"] = scale_linear(x["rating"], 0.5, 4.0)
    return reindex(x)


def load_jester(data_dir: Path, harmonize: bool = True):
    p = data_dir / "jester-data-1.csv"
    wide = pd.read_csv(p, header=None)
    values = wide.iloc[:, 1:].to_numpy(dtype=np.float32)
    mask = values != 99
    users, items = np.where(mask)
    ratings = values[users, items]
    if harmonize:
        ratings = 3.0 + 0.2 * ratings  # -10..10 -> 1..5
    x = pd.DataFrame({"user_id": users, "item_id": items, "rating": ratings})
    return reindex(x)


def run_dataset(name: str, data_dir: Path, out_dir: Path, native_jester: bool = False):
    if name == "BookCrossing":
        df, n_users, n_items = load_book_crossing(data_dir)
        seeds, batch_size, max_epochs, patience = SEEDS_5, 16384, 15, 3
    elif name == "FilmTrust":
        df, n_users, n_items = load_filmtrust(data_dir)
        seeds, batch_size, max_epochs, patience = SEEDS_5, 16384, 15, 3
    elif name == "Jester":
        df, n_users, n_items = load_jester(data_dir, harmonize=not native_jester)
        seeds, batch_size, max_epochs, patience = SEEDS_3, 131072, 10, 2
    else:
        raise ValueError(name)

    label = name + ("Native" if native_jester else "Scaled")
    train, val, test = make_split(df, split_seed=DEFAULT_SPLIT_SEED)
    orders = nested_orders(train, split_seed=DEFAULT_SPLIT_SEED)
    val_t = to_tensors(val)
    test_t = to_tensors(test)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "dataset": label,
        "raw_n": len(df),
        "users": n_users,
        "items": n_items,
        "rating_min": float(df.rating.min()),
        "rating_max": float(df.rating.max()),
        "train_n": len(train),
        "val_n": len(val),
        "test_n": len(test),
    }]).to_csv(out_dir / f"{label}_meta.csv", index=False)

    rows = []
    for frac in DEFAULT_RETENTIONS:
        sparse_train = sparsify(train, orders, frac, min_user_train=MIN_USER_TRAIN)
        actual_frac = len(sparse_train) / len(train)
        for seed in seeds:
            for model_name in MODELS:
                _, _, metrics = fit_model(
                    model_name,
                    sparse_train,
                    val_t,
                    test_t,
                    n_users,
                    n_items,
                    seed,
                    k=K,
                    lr=LR,
                    weight_decay=WEIGHT_DECAY,
                    batch_size=batch_size,
                    max_epochs=max_epochs,
                    patience=patience,
                )
                rows.append({
                    "dataset": label,
                    "retain_frac": frac,
                    "actual_frac": actual_frac,
                    "seed": seed,
                    "model": model_name,
                    "train_n": len(sparse_train),
                    **metrics,
                })
                print(label, frac, seed, model_name, metrics["rmse"], flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / f"{label}_long.csv", index=False)

    summary = (
        res.groupby(["dataset", "retain_frac", "actual_frac", "model"])
        .agg(
            n=("seed", "size"),
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_sd=("mae", "std"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / f"{label}_summary.csv", index=False)

    pivot = res.pivot_table(
        index=["dataset", "retain_frac", "actual_frac", "seed"],
        columns="model",
        values=["rmse", "mae"],
    ).reset_index()
    ablation_rows = []
    for (ds, frac, actual), g in pivot.groupby([("dataset", ""), ("retain_frac", ""), ("actual_frac", "")]):
        d_rmse = g[("rmse", "CosineMF")] - g[("rmse", "NormalizedMF")]
        d_mae = g[("mae", "CosineMF")] - g[("mae", "NormalizedMF")]
        relative = d_rmse / g[("rmse", "CosineMF")] * 100.0
        ablation_rows.append({
            "dataset": ds,
            "retain_frac": frac,
            "actual_frac": actual,
            "n": len(g),
            "delta_rmse_mean": d_rmse.mean(),
            "delta_rmse_sd": d_rmse.std(ddof=1),
            "relative_gain_pct": relative.mean(),
            "delta_mae_mean": d_mae.mean(),
            "delta_mae_sd": d_mae.std(ddof=1),
        })
    pd.DataFrame(ablation_rows).to_csv(out_dir / f"{label}_ablation.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["BookCrossing", "FilmTrust", "Jester", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "external_validation")
    parser.add_argument("--jester-native-scale", action="store_true", help="Run Jester on its native -10..10 scale as a sensitivity check.")
    args = parser.parse_args()

    names = ["BookCrossing", "FilmTrust", "Jester"] if args.dataset == "all" else [args.dataset]
    for name in names:
        run_dataset(name, args.data_dir, args.out_dir, native_jester=(name == "Jester" and args.jester_native_scale))


if __name__ == "__main__":
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    main()
