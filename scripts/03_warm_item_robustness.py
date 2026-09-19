from pathlib import Path
import argparse
import sys
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lfn.core import load_dataset, make_split, nested_orders, sparsify, to_tensors, fit_model, evaluate  # noqa: E402

MODELS = ("BiasOnly", "BiasMF", "CosineMF", "NormalizedMF")


def run(kind: str, data_dir: Path):
    df, n_users, n_items, cfg = load_dataset(kind, data_dir)
    base_train, val, test = make_split(df)
    orders = nested_orders(base_train)
    train = sparsify(base_train, orders, 0.10)
    seen_items = set(train.i.unique())
    warm_test = test[test.i.isin(seen_items)].reset_index(drop=True)
    val_t = to_tensors(val)
    warm_t = to_tensors(warm_test)
    rows = []

    for seed in cfg.seeds:
        for name in MODELS:
            model, mu, metrics = fit_model(
                name, train, val_t, warm_t, n_users, n_items, seed,
                k=32, lr=0.015, weight_decay=1e-5,
                batch_size=cfg.batch_size, max_epochs=cfg.max_epochs,
                patience=cfg.patience, device="cpu",
            )
            rmse, mae = evaluate(model, warm_t, mu, cfg.batch_size)
            rows.append([kind, seed, name, len(test), len(warm_test), len(test) - len(warm_test), rmse, mae])
            print(kind, seed, name, f"rmse={rmse:.4f}", flush=True)
    return pd.DataFrame(rows, columns=[
        "dataset", "seed", "model", "test_n_all", "test_n_warm", "cold_item_test_n", "rmse_warm", "mae_warm",
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))

    result = pd.concat([run("100K", args.data_dir), run("1M", args.data_dir)], ignore_index=True)
    result.to_csv(args.out_dir / "warm_item_10pct_robustness.csv", index=False)
    summary = (
        result.groupby(["dataset", "model"])
        .agg(
            n=("seed", "size"),
            rmse_mean=("rmse_warm", "mean"), rmse_sd=("rmse_warm", "std"),
            mae_mean=("mae_warm", "mean"), mae_sd=("mae_warm", "std"),
            test_n_warm=("test_n_warm", "first"), cold_item_test_n=("cold_item_test_n", "first"),
        )
        .reset_index()
    )
    summary.to_csv(args.out_dir / "warm_item_10pct_robustness_summary.csv", index=False)


if __name__ == "__main__":
    main()
