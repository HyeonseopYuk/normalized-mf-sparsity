from pathlib import Path
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    r100 = pd.read_csv(args.out_dir / "ML100K_validated_long.csv")
    r1m = pd.read_csv(args.out_dir / "ML1M_validated_long.csv")

    for name, df in (("ML100K", r100), ("ML1M", r1m)):
        summary = (
            df.groupby(["retain_frac", "model"])
            .agg(
                n=("seed", "size"),
                rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
                mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
                best_epoch_mean=("best_epoch", "mean"), runtime_mean=("train_seconds", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(args.out_dir / f"{name}_validated_summary.csv", index=False)

    rows = []
    for dataset, df in (("MovieLens 100K", r100), ("MovieLens 1M", r1m)):
        for frac in sorted(df.retain_frac.unique(), reverse=True):
            pivot = df[df.retain_frac == frac].pivot(index="seed", columns="model", values=["rmse", "mae"])
            for metric in ("rmse", "mae"):
                diff = pivot[(metric, "CosineMF")] - pivot[(metric, "NormalizedMF")]
                baseline = pivot[(metric, "CosineMF")]
                relative = diff / baseline * 100.0
                rows.append([
                    dataset, frac, metric, diff.mean(), diff.std(ddof=1), len(diff),
                    relative.mean(), relative.std(ddof=1),
                ])
    ablation = pd.DataFrame(rows, columns=[
        "dataset", "retain_frac", "metric", "abs_gain_mean", "abs_gain_sd", "n_seeds",
        "relative_gain_pct_mean", "relative_gain_pct_sd",
    ])
    ablation.to_csv(args.out_dir / "direct_ablation_cosine_vs_normalized.csv", index=False)


if __name__ == "__main__":
    main()
