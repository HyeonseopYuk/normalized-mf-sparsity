from pathlib import Path
import argparse
import sys
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lfn.core import (  # noqa: E402
    DEFAULT_RETENTIONS,
    load_dataset,
    make_split,
    nested_orders,
    sparsify,
    to_tensors,
    fit_model,
)

MODELS = ("BiasOnly", "BiasMF", "CosineMF", "NormalizedMF")


def run_dataset(kind: str, data_dir: Path, out_dir: Path, device: str = "cpu"):
    df, n_users, n_items, cfg = load_dataset(kind, data_dir)
    base_train, val, test = make_split(df)
    orders = nested_orders(base_train)
    val_t = to_tensors(val, device)
    test_t = to_tensors(test, device)
    rows = []

    for frac in DEFAULT_RETENTIONS:
        train = sparsify(base_train, orders, frac)
        print(f"{kind} retention={frac:.2f} train_n={len(train):,}", flush=True)
        for seed in cfg.seeds:
            line = []
            for model_name in MODELS:
                _, _, m = fit_model(
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
                rows.append([
                    kind, frac, seed, model_name, len(train),
                    m["rmse"], m["mae"], m["best_epoch"], m["train_seconds"], m["val_rmse"],
                ])
                line.append(f"{model_name}:{m['rmse']:.4f}")
            print(seed, " | ".join(line), flush=True)
            pd.DataFrame(rows, columns=[
                "dataset", "retain_frac", "seed", "model", "train_n", "rmse", "mae",
                "best_epoch", "train_seconds", "val_rmse",
            ]).to_csv(out_dir / f"ML{kind}_validated_long.csv", index=False)

    result = pd.DataFrame(rows, columns=[
        "dataset", "retain_frac", "seed", "model", "train_n", "rmse", "mae",
        "best_epoch", "train_seconds", "val_rmse",
    ])
    summary = (
        result.groupby(["retain_frac", "model"])
        .agg(
            n=("seed", "size"),
            rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
            mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
            best_epoch_mean=("best_epoch", "mean"), runtime_mean=("train_seconds", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / f"ML{kind}_validated_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["100K", "1M", "both"], default="both")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    if args.dataset in ("100K", "both"):
        run_dataset("100K", args.data_dir, args.out_dir, args.device)
    if args.dataset in ("1M", "both"):
        run_dataset("1M", args.data_dir, args.out_dir, args.device)


if __name__ == "__main__":
    main()
