from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lfn.core import (  # noqa: E402
    DEFAULT_RETENTIONS,
    fit_model,
    load_dataset,
    make_split,
    nested_orders,
    sparsify,
    to_tensors,
)


@torch.no_grad()
def predict(model, mu: float, df: pd.DataFrame, batch_size: int, device: str = "cpu") -> np.ndarray:
    u, i, y = to_tensors(df, device)
    preds = []
    model.eval()
    for start in range(0, len(y), batch_size):
        end = min(start + batch_size, len(y))
        preds.append(model(u[start:end], i[start:end], mu).detach().cpu().numpy())
    return np.concatenate(preds)


def cluster_sse(test: pd.DataFrame, y: np.ndarray, pred: np.ndarray, n_users: int):
    err2 = (pred - y) ** 2
    sse = np.bincount(test.u.to_numpy(), weights=err2, minlength=n_users).astype(np.float64)
    n = np.bincount(test.u.to_numpy(), minlength=n_users).astype(np.float64)
    return sse, n


def bootstrap_delta(
    cos_sse: np.ndarray,
    norm_sse: np.ndarray,
    n_user: np.ndarray,
    *,
    reps: int,
    seed: int,
    chunk: int = 250,
) -> np.ndarray:
    """User-clustered paired bootstrap for mean paired delta RMSE across fixed seeds."""
    rng = np.random.default_rng(seed)
    n_users = len(n_user)
    probs = np.full(n_users, 1.0 / n_users)
    out = np.empty(reps, dtype=np.float64)
    pos = 0
    while pos < reps:
        batch = min(chunk, reps - pos)
        weights = rng.multinomial(n_users, probs, size=batch).astype(np.float64)
        denom = weights @ n_user
        cos_rmse = np.sqrt((weights @ cos_sse.T) / denom[:, None])
        norm_rmse = np.sqrt((weights @ norm_sse.T) / denom[:, None])
        out[pos : pos + batch] = (cos_rmse - norm_rmse).mean(axis=1)
        pos += batch
    return out


def run_dataset(kind: str, data_dir: Path, reps: int, bootstrap_seed: int, device: str):
    df, n_users, n_items, cfg = load_dataset(kind, data_dir)
    base_train, val, test = make_split(df)
    orders = nested_orders(base_train)
    val_t = to_tensors(val, device)
    test_t = to_tensors(test, device)
    y = test.rating.to_numpy(dtype=np.float64)
    n_user = np.bincount(test.u.to_numpy(), minlength=n_users).astype(np.float64)

    rows = []
    for frac in DEFAULT_RETENTIONS:
        train = sparsify(base_train, orders, frac)
        cosine_sse = []
        normalized_sse = []
        seed_deltas = []

        for seed in cfg.seeds:
            per_model = {}
            for model_name in ("CosineMF", "NormalizedMF"):
                model, mu, metrics = fit_model(
                    model_name,
                    train,
                    val_t,
                    test_t,
                    n_users,
                    n_items,
                    seed,
                    k=32,
                    lr=0.015,
                    weight_decay=1e-5,
                    batch_size=cfg.batch_size,
                    max_epochs=cfg.max_epochs,
                    patience=cfg.patience,
                    device=device,
                )
                pred = predict(model, mu, test, cfg.batch_size, device)
                sse, _ = cluster_sse(test, y, pred, n_users)
                per_model[model_name] = (sse, metrics)

            cosine_sse.append(per_model["CosineMF"][0])
            normalized_sse.append(per_model["NormalizedMF"][0])
            seed_deltas.append(
                per_model["CosineMF"][1]["rmse"] - per_model["NormalizedMF"][1]["rmse"]
            )

        cosine_sse = np.stack(cosine_sse)
        normalized_sse = np.stack(normalized_sse)
        condition_seed = bootstrap_seed + int(frac * 1000) + (0 if kind == "100K" else 10000)
        boot = bootstrap_delta(
            cosine_sse,
            normalized_sse,
            n_user,
            reps=reps,
            seed=condition_seed,
        )
        lo, hi = np.quantile(boot, [0.025, 0.975])
        rows.append(
            {
                "dataset": kind,
                "retention": frac,
                "n_users_test": int((n_user > 0).sum()),
                "n_test": int(n_user.sum()),
                "n_seeds": len(cfg.seeds),
                "delta_rmse_mean": float(np.mean(seed_deltas)),
                "ci95_low": float(lo),
                "ci95_high": float(hi),
                "boot_median": float(np.median(boot)),
                "bootstrap_reps": reps,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["100K", "1M", "both"], default="both")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-file", type=Path, default=ROOT / "results" / "user_cluster_bootstrap_ci.csv")
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260922)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))

    kinds = ["100K", "1M"] if args.dataset == "both" else [args.dataset]
    rows = []
    for kind in kinds:
        print(f"Bootstrap dataset: {kind}", flush=True)
        rows.extend(run_dataset(kind, args.data_dir, args.bootstrap_reps, args.bootstrap_seed, args.device))
        pd.DataFrame(rows).to_csv(args.out_file, index=False)

    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
