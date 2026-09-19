from pathlib import Path
import argparse
import sys
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lfn.core import load_dataset, make_split, nested_orders, sparsify, to_tensors, fit_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))

    df, n_users, n_items, cfg = load_dataset("100K", args.data_dir)
    base_train, val, test = make_split(df)
    orders = nested_orders(base_train)
    val_t, test_t = to_tensors(val), to_tensors(test)

    rows = []
    for frac in (1.0, 0.5, 0.1):
        train = sparsify(base_train, orders, frac)
        for k in (16, 32, 64):
            for wd in (1e-5, 1e-4, 1e-3):
                for seed in (11, 22, 33):
                    values = {}
                    for name in ("CosineMF", "NormalizedMF"):
                        _, _, m = fit_model(
                            name, train, val_t, test_t, n_users, n_items, seed,
                            k=k, lr=0.015, weight_decay=wd,
                            batch_size=cfg.batch_size, max_epochs=cfg.max_epochs,
                            patience=cfg.patience, device="cpu",
                        )
                        values[name] = m["rmse"]
                    gain = values["CosineMF"] - values["NormalizedMF"]
                    rows.append([frac, k, wd, seed, values["CosineMF"], values["NormalizedMF"], gain])
                    print(frac, k, wd, seed, f"gain={gain:+.5f}", flush=True)
                    pd.DataFrame(rows, columns=[
                        "retain_frac", "K", "weight_decay", "seed",
                        "cosine_rmse", "normalized_rmse", "gain_cosine_minus_norm",
                    ]).to_csv(args.out_dir / "ML100K_cosine_norm_sensitivity_long.csv", index=False)

    result = pd.DataFrame(rows, columns=[
        "retain_frac", "K", "weight_decay", "seed",
        "cosine_rmse", "normalized_rmse", "gain_cosine_minus_norm",
    ])
    summary = (
        result.groupby(["retain_frac", "K", "weight_decay"])
        .agg(gain_mean=("gain_cosine_minus_norm", "mean"), gain_sd=("gain_cosine_minus_norm", "std"))
        .reset_index()
    )
    summary.to_csv(args.out_dir / "ML100K_cosine_norm_sensitivity_summary.csv", index=False)


if __name__ == "__main__":
    main()
