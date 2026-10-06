from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lfn.core import BiasMF, CosineMF, NormalizedMF, load_dataset, make_split, nested_orders, sparsify, to_tensors  # noqa: E402

MODELS = ("BiasMF", "CosineMF", "NormalizedMF", "NCF")
DEFAULT_RETENTIONS = (1.0, 0.5, 0.1)
EVAL_SEEDS = (11, 22, 33, 44, 55)
TUNE_SEED = 707


class NCFRegressor(nn.Module):
    """NeuMF-style explicit-rating baseline with user/item bias terms."""

    def __init__(self, n_users: int, n_items: int, k: int):
        super().__init__()
        self.user_gmf = nn.Embedding(n_users, k)
        self.item_gmf = nn.Embedding(n_items, k)
        self.user_mlp = nn.Embedding(n_users, k)
        self.item_mlp = nn.Embedding(n_items, k)
        self.ub = nn.Embedding(n_users, 1)
        self.ib = nn.Embedding(n_items, 1)
        h1 = max(32, 2 * k)
        h2 = max(16, k)
        self.mlp = nn.Sequential(
            nn.Linear(2 * k, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        self.out = nn.Linear(k + h2, 1)
        for emb in (self.user_gmf, self.item_gmf, self.user_mlp, self.item_mlp):
            nn.init.normal_(emb.weight, std=0.05)
        nn.init.zeros_(self.ub.weight)
        nn.init.zeros_(self.ib.weight)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, u, i, mu):
        gmf = self.user_gmf(u) * self.item_gmf(i)
        mlp = self.mlp(torch.cat([self.user_mlp(u), self.item_mlp(i)], dim=1))
        interaction = self.out(torch.cat([gmf, mlp], dim=1)).squeeze(1)
        return mu + self.ub(u).squeeze(1) + self.ib(i).squeeze(1) + interaction


def make_candidate(name: str, n_users: int, n_items: int, k: int):
    if name == "BiasMF":
        return BiasMF(n_users, n_items, k)
    if name == "CosineMF":
        return CosineMF(n_users, n_items, k)
    if name == "NormalizedMF":
        return NormalizedMF(n_users, n_items, k)
    if name == "NCF":
        return NCFRegressor(n_users, n_items, k)
    raise ValueError(name)


@torch.no_grad()
def predict(model, tensors, mu: float, batch_size: int):
    u, i, y = tensors
    preds = []
    model.eval()
    for start in range(0, len(y), batch_size):
        end = min(start + batch_size, len(y))
        preds.append(model(u[start:end], i[start:end], mu).detach().cpu())
    return torch.cat(preds).numpy()


def metric_values(y: np.ndarray, pred: np.ndarray):
    d = pred - y
    return float(np.sqrt(np.mean(d * d))), float(np.mean(np.abs(d)))


def fit_one(name, train_df, val_df, test_df, n_users, n_items, seed, *,
            k, lr, weight_decay, batch_size, max_epochs, patience, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    train_t = to_tensors(train_df, device)
    val_t = to_tensors(val_df, device)
    test_t = to_tensors(test_df, device) if test_df is not None else None
    mu = float(train_df.rating.mean())
    model = make_candidate(name, n_users, n_items, k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    u, i, y = train_t
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        gen = torch.Generator().manual_seed(seed * 1000 + epoch)
        order = torch.randperm(len(y), generator=gen).to(device)
        for start in range(0, len(y), batch_size):
            idx = order[start:start + batch_size]
            pred = model(u[idx], i[idx], mu)
            loss = ((pred - y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_pred = predict(model, val_t, mu, batch_size)
        val_rmse, _ = metric_values(val_df.rating.to_numpy(np.float32), val_pred)
        if val_rmse < best - 1e-5:
            best = val_rmse
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        raise RuntimeError("No validation state retained")
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    elapsed = time.perf_counter() - started
    out = {"val_rmse": best, "best_epoch": best_epoch, "train_seconds": elapsed}
    if test_t is not None:
        p = predict(model, test_t, mu, batch_size)
        rmse, mae = metric_values(test_df.rating.to_numpy(np.float32), p)
        out.update({"rmse": rmse, "mae": mae, "pred": p})
    return out


def tuning_grid():
    return [{"k": k, "lr": lr, "weight_decay": wd}
            for k, lr, wd in itertools.product((16, 32), (0.005, 0.015), (1e-5, 1e-4))]


def holm_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * pvals[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def user_rmse(test_df: pd.DataFrame, preds_by_seed):
    pred_mean = np.mean(np.stack(preds_by_seed, axis=0), axis=0)
    tmp = test_df[["u", "rating"]].copy()
    tmp["sq"] = (pred_mean - tmp.rating.to_numpy()) ** 2
    return tmp.groupby("u").sq.mean().pow(0.5)


def run(kind: str, data_dir: Path, out_dir: Path, device: str):
    df, n_users, n_items, cfg = load_dataset(kind, data_dir)
    base_train, val, test = make_split(df)
    orders = nested_orders(base_train)
    grid = tuning_grid()
    tuning_rows, eval_rows, stats_rows = [], [], []
    pred_store, selected = {}, {}

    for frac in DEFAULT_RETENTIONS:
        train = sparsify(base_train, orders, frac)
        print(f"{kind} retention={frac:.2f} train={len(train):,}", flush=True)

        for model_name in MODELS:
            trials = []
            for hp in grid:
                m = fit_one(model_name, train, val, None, n_users, n_items, TUNE_SEED,
                            batch_size=cfg.batch_size, max_epochs=cfg.max_epochs,
                            patience=cfg.patience, device=device, **hp)
                row = {"dataset": kind, "retain_frac": frac, "model": model_name,
                       "tune_seed": TUNE_SEED, **hp, "val_rmse": m["val_rmse"],
                       "best_epoch": m["best_epoch"], "train_seconds": m["train_seconds"]}
                tuning_rows.append(row)
                trials.append(row)
                print(model_name, hp, f"val={m['val_rmse']:.5f}", flush=True)

            best = min(trials, key=lambda x: (x["val_rmse"], x["k"], x["lr"], x["weight_decay"]))
            hp = {key: best[key] for key in ("k", "lr", "weight_decay")}
            selected[(frac, model_name)] = hp
            print("SELECT", kind, frac, model_name, hp, f"val={best['val_rmse']:.5f}", flush=True)

            preds = []
            for seed in EVAL_SEEDS:
                m = fit_one(model_name, train, val, test, n_users, n_items, seed,
                            batch_size=cfg.batch_size, max_epochs=cfg.max_epochs,
                            patience=cfg.patience, device=device, **hp)
                preds.append(m.pop("pred"))
                eval_rows.append({"dataset": kind, "retain_frac": frac, "model": model_name,
                                  "seed": seed, **hp, **m})
                print("EVAL", model_name, seed, f"rmse={m['rmse']:.5f}", flush=True)
            pred_store[(frac, model_name)] = preds

        pd.DataFrame(tuning_rows).to_csv(out_dir / f"ML{kind}_model_specific_tuning.csv", index=False)
        pd.DataFrame(eval_rows).to_csv(out_dir / f"ML{kind}_tuned_evaluation_long.csv", index=False)

    eval_df = pd.DataFrame(eval_rows)
    summary = (eval_df.groupby(["dataset", "retain_frac", "model", "k", "lr", "weight_decay"], as_index=False)
               .agg(n=("seed", "size"), rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
                    mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
                    val_rmse_mean=("val_rmse", "mean")))
    summary.to_csv(out_dir / f"ML{kind}_tuned_evaluation_summary.csv", index=False)

    for frac in DEFAULT_RETENTIONS:
        norm_user = user_rmse(test, pred_store[(frac, "NormalizedMF")])
        raw_p, local = [], []
        for comparator in ("CosineMF", "BiasMF", "NCF"):
            comp_user = user_rmse(test, pred_store[(frac, comparator)])
            common = norm_user.index.intersection(comp_user.index)
            diff = comp_user.loc[common].to_numpy() - norm_user.loc[common].to_numpy()
            stat, p = wilcoxon(diff, alternative="two-sided", zero_method="wilcox", method="auto")
            raw_p.append(float(p))
            local.append({"dataset": kind, "retain_frac": frac,
                          "comparison": f"{comparator} - NormalizedMF",
                          "n_users": len(common), "mean_user_rmse_diff": float(np.mean(diff)),
                          "median_user_rmse_diff": float(np.median(diff)),
                          "wilcoxon_stat": float(stat), "p_raw": float(p)})
        adj = holm_adjust(raw_p)
        for row, p_adj in zip(local, adj):
            row["p_holm"] = float(p_adj)
            stats_rows.append(row)

    pd.DataFrame(stats_rows).to_csv(out_dir / f"ML{kind}_user_level_wilcoxon.csv", index=False)
    with open(out_dir / f"ML{kind}_selected_hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump({f"retention={frac:.2f}|{model}": hp for (frac, model), hp in selected.items()},
                  f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["100K", "1M", "both"], default="both")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "resubmission")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    if args.dataset in ("100K", "both"):
        run("100K", args.data_dir, args.out_dir, args.device)
    if args.dataset in ("1M", "both"):
        run("1M", args.data_dir, args.out_dir, args.device)


if __name__ == "__main__":
    main()
