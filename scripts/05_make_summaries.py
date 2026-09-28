from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _primary_ablation(df: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    rows = []
    for frac in sorted(df.retain_frac.unique(), reverse=True):
        g = df[df.retain_frac == frac]
        pivot = g.pivot(index="seed", columns="model", values=["rmse", "mae"])
        d_rmse = pivot[("rmse", "CosineMF")] - pivot[("rmse", "NormalizedMF")]
        d_mae = pivot[("mae", "CosineMF")] - pivot[("mae", "NormalizedMF")]
        rel = d_rmse / pivot[("rmse", "CosineMF")] * 100.0
        rows.append({
            "dataset": dataset_label,
            "nominal_retention": frac,
            "actual_retention": g.train_n.iloc[0] / df[df.retain_frac == 1.0].train_n.iloc[0],
            "n_seeds": len(d_rmse),
            "delta_rmse": d_rmse.mean(),
            "delta_rmse_sd": d_rmse.std(ddof=1),
            "relative_rmse_gain_pct": rel.mean(),
            "delta_mae": d_mae.mean(),
            "delta_mae_sd": d_mae.std(ddof=1),
        })
    return pd.DataFrame(rows)


def _external_ablation(path: Path, dataset_label: str) -> pd.DataFrame:
    x = pd.read_csv(path)
    return pd.DataFrame({
        "dataset": dataset_label,
        "nominal_retention": x["retain_frac"],
        "actual_retention": x["actual_frac"],
        "n_seeds": x["n"],
        "delta_rmse": x["delta_rmse_mean"],
        "delta_rmse_sd": x["delta_rmse_sd"],
        "relative_rmse_gain_pct": x["relative_gain_pct"],
        "delta_mae": x["delta_mae_mean"],
        "delta_mae_sd": x["delta_mae_sd"],
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    out = args.results_dir
    ext = out / "external_validation"

    r100 = pd.read_csv(out / "ML100K_validated_long.csv")
    r1m = pd.read_csv(out / "ML1M_validated_long.csv")

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
        summary.to_csv(out / f"{name}_validated_summary.csv", index=False)

    # Primary two-dataset paired-ablation summary.
    paired_rows = []
    for dataset, df in (("MovieLens 100K", r100), ("MovieLens 1M", r1m)):
        for frac in sorted(df.retain_frac.unique(), reverse=True):
            pivot = df[df.retain_frac == frac].pivot(index="seed", columns="model", values=["rmse", "mae"])
            for metric in ("rmse", "mae"):
                diff = pivot[(metric, "CosineMF")] - pivot[(metric, "NormalizedMF")]
                baseline = pivot[(metric, "CosineMF")]
                relative = diff / baseline * 100.0
                paired_rows.append([
                    dataset, frac, metric, diff.mean(), diff.std(ddof=1), len(diff),
                    relative.mean(), relative.std(ddof=1),
                ])
    pd.DataFrame(paired_rows, columns=[
        "dataset", "retain_frac", "metric", "abs_gain_mean", "abs_gain_sd", "n_seeds",
        "relative_gain_pct_mean", "relative_gain_pct_sd",
    ]).to_csv(out / "direct_ablation_cosine_vs_normalized.csv", index=False)

    # Five-dataset table used by manuscript Fig. 2 and Supplementary Fig. S1.
    parts = [
        _primary_ablation(r100, "MovieLens100K"),
        _primary_ablation(r1m, "MovieLens1M"),
    ]
    external_specs = [
        (ext / "BookCrossingScaled_ablation.csv", "BookCrossing"),
        (ext / "FilmTrustScaled_ablation.csv", "FilmTrust"),
        (ext / "JesterScaled_ablation.csv", "Jester"),
    ]
    missing = []
    for path, label in external_specs:
        if path.exists():
            parts.append(_external_ablation(path, label))
        else:
            missing.append(path.name)

    combined = pd.concat(parts, ignore_index=True)
    combined.to_csv(out / "five_dataset_direct_ablation.csv", index=False)

    retention = combined[["dataset", "nominal_retention", "actual_retention"]].copy()
    retention["dataset"] = retention["dataset"].replace({
        "MovieLens100K": "MovieLens 100K",
        "MovieLens1M": "MovieLens 1M",
        "BookCrossing": "Book-Crossing",
    })
    retention.to_csv(out / "retention_metadata.csv", index=False)

    if missing:
        print("Note: external validation files were not found for: " + ", ".join(missing))
        print("Run scripts/07_external_validation.py --dataset all, then rerun this script for all five datasets.")


if __name__ == "__main__":
    main()
