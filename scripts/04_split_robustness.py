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
    rows = []
    for split_seed in (20260918, 20260919, 20260920):
        base_train, val, test = make_split(df, split_seed=split_seed)
        orders = nested_orders(base_train, split_seed=split_seed)
        val_t, test_t = to_tensors(val), to_tensors(test)
        for frac in (1.0, 0.5, 0.1):
            train = sparsify(base_train, orders, frac)
            vals = {}
            for name in ("CosineMF", "NormalizedMF"):
                _, _, m = fit_model(
                    name, train, val_t, test_t, n_users, n_items, 11,
                    k=32, lr=0.015, weight_decay=1e-5,
                    batch_size=cfg.batch_size, max_epochs=cfg.max_epochs,
                    patience=cfg.patience, device="cpu",
                )
                vals[name] = m["rmse"]
            gain = vals["CosineMF"] - vals["NormalizedMF"]
            rows.append([split_seed, frac, vals["CosineMF"], vals["NormalizedMF"], gain])
            print(split_seed, frac, f"gain={gain:+.5f}", flush=True)

    result = pd.DataFrame(rows, columns=[
        "split_seed", "retain_frac", "cosine_rmse", "normalized_rmse", "gain_cosine_minus_norm",
    ])
    result.to_csv(args.out_dir / "ML100K_split_robustness.csv", index=False)


if __name__ == "__main__":
    main()
