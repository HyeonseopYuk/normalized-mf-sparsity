# Latent-Factor Normalization under Controlled Rating Sparsity

Reproducibility repository for the manuscript on latent-factor normalization in explicit-rating matrix-factorization recommenders under controlled sparsity.

## What is reproduced

The repository implements the final manuscript design:

- Primary datasets: MovieLens 100K and MovieLens 1M
- Cross-domain external validation: Book-Crossing, FilmTrust, and Jester
- User-stratified 80/10/10 train/validation/test split
- Fixed validation/test sets
- Nested training-retention levels: 100%, 75%, 50%, 25%, 10%
- Minimum of 5 retained training ratings per user
- Models: BiasOnly, BiasMF, CosineMF, NormalizedMF
- Metrics: RMSE and MAE
- MovieLens 100K: 10 initialization seeds
- MovieLens 1M: 5 initialization seeds
- Sensitivity analysis over latent dimension and weight decay
- Warm-item-only robustness check at 10% retention
- Split-robustness analysis on MovieLens 100K
- Final manuscript figures
- Cross-domain rating-scale harmonization and direct-ablation summaries

The repeated initialization seeds are used to assess **optimization stability**, not population-sampling uncertainty.

## Exact software versions used for the finalized analysis

| Package | Version |
|---|---:|
| Python | 3.13.5 |
| NumPy | 2.3.5 |
| pandas | 2.2.3 |
| SciPy | 1.17.0 |
| PyTorch | 2.10.0+cpu |
| Matplotlib | 3.10.8 |
| openpyxl | 3.1.5 |
| python-docx | 1.2.0 |

See [`ENVIRONMENT.md`](ENVIRONMENT.md) for reproducibility notes.

## Setup

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data

The MovieLens datasets are not redistributed in this repository. See [`data/README.md`](data/README.md).

Expected files:

```text
data/
├── ml-100k.zip
└── ratings.dat
```

Alternatively, `ml-1m.zip` may be placed in `data/`; the loader will extract `ml-1m/ratings.dat` automatically.

## Run the full analysis

```bash
python scripts/01_main_experiments.py --dataset both
python scripts/02_sensitivity_analysis.py
python scripts/03_warm_item_robustness.py
python scripts/04_split_robustness.py
python scripts/05_make_summaries.py
python scripts/06_make_final_figures.py
python scripts/07_external_validation.py --dataset all
```

Results are written to `results/` and publication figures to `figures/`.

### Run one dataset only

```bash
python scripts/01_main_experiments.py --dataset 100K
python scripts/01_main_experiments.py --dataset 1M
```

## External-validation protocol

Book-Crossing, FilmTrust, and Jester are used as validation benchmarks rather than equal-depth replacements for the two primary MovieLens experiments. Their explicit ratings are mapped to a common 1-5 scale before fitting. Book-Crossing and FilmTrust retain only users with at least seven observations so that the split can allocate at least five training ratings plus validation and test observations. Jester uses 99 as the missing-value code.

A native-scale Jester sensitivity check can be run with:

```bash
python scripts/07_external_validation.py --dataset Jester --jester-native-scale
```

## Core model definitions

### BiasMF

`mu + b_u + b_i + p_u^T q_i`

### CosineMF

The user/item vectors are L2-normalized and their cosine interaction is multiplied by one learned positive global scale.

### NormalizedMF

The user/item vectors are L2-normalized, but each latent dimension receives its own learned positive scale. This is the direct ablation against CosineMF used in the manuscript.

## Main fixed hyperparameters

| Setting | Value |
|---|---:|
| Latent dimension K | 32 |
| Adam learning rate | 0.015 |
| Weight decay | 1e-5 |
| Minimum retained ratings/user | 5 |
| Split seed | 20260918 |
| 100K batch size | 16,384 |
| 100K max epochs / patience | 12 / 3 |
| 1M batch size | 131,072 |
| 1M max epochs / patience | 8 / 2 |

Sensitivity analysis additionally uses `K in {16, 32, 64}` and weight decay in `{1e-5, 1e-4, 1e-3}`.

## Repository structure

```text
.
├── README.md
├── README_KR.md
├── ENVIRONMENT.md
├── VERSION
├── requirements.txt
├── src/lfn/
│   ├── __init__.py
│   └── core.py
├── scripts/
│   ├── 01_main_experiments.py
│   ├── 02_sensitivity_analysis.py
│   ├── 03_warm_item_robustness.py
│   ├── 04_split_robustness.py
│   ├── 05_make_summaries.py
│   ├── 06_make_final_figures.py
│   └── 07_external_validation.py
├── data/
├── results/
├── figures/
└── .github/workflows/python-smoke.yml
```

## Reproducibility note

PyTorch seed control substantially reduces variation, but bit-for-bit identical floating-point results are not guaranteed across hardware, BLAS backends, CPU/GPU builds, or operating systems. The expected manuscript-level pattern should be assessed from the reported means and direct paired model differences rather than requiring identical final decimal digits.

## License

This repository is released under the [MIT License](LICENSE).
