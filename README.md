# Global versus Dimension-Specific Scaling in Normalized Matrix Factorization under Rating Sparsity

Reproducibility repository for the manuscript **"Global versus Dimension-Specific Scaling in Normalized Matrix Factorization under Rating Sparsity"**.

This repository contains the analysis code and manuscript-level processed outputs for the controlled comparison between a single global cosine scale (CosineMF) and positive dimension-specific scaling after L2 normalization (NormalizedMF).

## Study design reproduced here

- Primary controlled benchmarks: MovieLens 100K and MovieLens 1M
- Multi-domain replication: Book-Crossing, FilmTrust, and Jester
- User-stratified approximately 80/10/10 train/validation/test split
- Fixed validation and test sets within each split
- Nested training-retention levels: 100%, 75%, 50%, 25%, and 10%
- Minimum of 5 retained training ratings per user
- Models: BiasOnly, BiasMF, CosineMF, and NormalizedMF
- Metrics: RMSE and MAE
- MovieLens 100K: 10 initialization seeds
- MovieLens 1M: 5 initialization seeds
- Book-Crossing and FilmTrust: 5 initialization seeds
- Jester: 3 initialization seeds
- Sensitivity analysis over latent dimension and weight decay
- Split-robustness analysis on MovieLens 100K
- Warm-item-only robustness analysis at 10% retention
- User-clustered paired bootstrap for the primary MovieLens ablation
- Jester native-rating-scale sensitivity analysis

Repeated initialization seeds quantify **optimization variability**. They are not treated as independent population samples. The user-clustered bootstrap quantifies test-user sampling variation conditional on the fitted models and benchmark design.

## Core model comparison

### BiasMF

`mu + b_u + b_i + p_u^T q_i`

### CosineMF

User and item latent vectors are L2-normalized, and the normalized interaction is multiplied by one learned positive global scale.

### NormalizedMF

User and item latent vectors are L2-normalized, but each latent dimension receives a separate learned positive scale. This is the direct ablation against CosineMF used in the manuscript.

## Software environment

The finalized analysis used:

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

See [`ENVIRONMENT.md`](ENVIRONMENT.md) for notes on reproducibility across platforms.

## Setup

```bash
python -m venv .venv

# Git Bash / Linux / macOS
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data

Raw third-party datasets are **not redistributed** in this repository. Place the required files under `data/` before running the analysis. See [`data/README.md`](data/README.md) for filenames and preprocessing details.

## Recommended reproduction order

The following order recreates the manuscript-level analyses from raw data:

```bash
# 1. Primary MovieLens experiments
python scripts/01_main_experiments.py --dataset both

# 2. Sensitivity analyses
python scripts/02_sensitivity_analysis.py

# 3. Warm-item robustness
python scripts/03_warm_item_robustness.py

# 4. Alternative split seeds
python scripts/04_split_robustness.py

# 5. Multi-domain replication on the harmonized 1-5 scale
python scripts/07_external_validation.py --dataset all

# 6. Native-scale Jester sensitivity check
python scripts/07_external_validation.py --dataset Jester --jester-native-scale

# 7. Manuscript-level summary and five-dataset ablation tables
python scripts/05_make_summaries.py

# 8. User-clustered paired bootstrap for MovieLens
python scripts/08_user_cluster_bootstrap.py --dataset both --bootstrap-reps 10000

# 9. Publication figures
python scripts/06_make_final_figures.py
```

The complete experiment set can be computationally expensive, particularly MovieLens 1M, Jester, and the bootstrap analysis.

## Main fixed hyperparameters

| Setting | Value |
|---|---:|
| Latent dimension K | 32 |
| Adam learning rate | 0.015 |
| Weight decay | 1e-5 |
| Minimum retained ratings/user | 5 |
| Primary split seed | 20260918 |
| MovieLens 100K batch size | 16,384 |
| MovieLens 100K max epochs / patience | 12 / 3 |
| MovieLens 1M batch size | 131,072 |
| MovieLens 1M max epochs / patience | 8 / 2 |

Sensitivity analysis additionally uses `K in {16, 32, 64}` and weight decay in `{1e-5, 1e-4, 1e-3}`.

## Repository structure

```text
.
├── README.md
├── README_KR.md
├── ENVIRONMENT.md
├── VERSION
├── LICENSE
├── CITATION.cff
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
│   ├── 07_external_validation.py
│   └── 08_user_cluster_bootstrap.py
├── data/
│   └── README.md
├── results/
│   ├── README.md
│   ├── manuscript-level CSV summaries
│   └── external_validation/
├── figures/
│   └── README.md
├── supplementary/
│   ├── Supplementary_Material.docx
│   ├── Supplementary_tables.zip
│   └── README.md
└── .github/workflows/python-smoke.yml
```

## Manuscript-level processed outputs

The repository includes the processed CSV summaries used to support the manuscript and supplement, including:

- `ML100K_validated_summary.csv`
- `ML1M_validated_summary.csv`
- `ML100K_cosine_norm_sensitivity_summary.csv`
- `five_dataset_direct_ablation.csv`
- `retention_metadata.csv`
- `user_cluster_bootstrap_ci.csv`
- external-validation summary and ablation tables under `results/external_validation/`

Long per-seed outputs are regenerable and are not required to remain under version control.

## Interpretation of the bootstrap intervals

The user-clustered paired bootstrap resamples test users while keeping all ratings from a sampled user together. For each bootstrap resample, paired CosineMF and NormalizedMF RMSE values are recomputed for the fixed trained seed pairs and then averaged across seeds. The resulting percentile intervals describe **test-user sampling variation conditional on the evaluated benchmark datasets and fitted models**. They are not population-level confidence intervals.

## Reproducibility note

Fixed seeds substantially reduce stochastic variation, but bit-for-bit identical floating-point output is not guaranteed across hardware, operating systems, BLAS backends, or PyTorch builds. Reproduction should focus on manuscript-level means and paired model differences rather than exact agreement in the last decimal place.

## License and third-party data

The source code in this repository is released under the [MIT License](LICENSE). Third-party datasets are not included and remain subject to their respective licenses and terms of use.
