# Global versus Dimension-Specific Scaling in Normalized Matrix Factorization under Rating Sparsity

Reproducibility repository for the manuscript **"Global versus Dimension-Specific Scaling in Normalized Matrix Factorization under Rating Sparsity"**.

This repository contains the analysis code and manuscript-level processed outputs for a controlled comparison between:

- a single positive global scale after L2 normalization (**CosineMF**), and
- positive dimension-specific scales after L2 normalization (**NormalizedMF**).

The repository is intended to reproduce the primary experiments, robustness analyses, multi-domain replications, user-clustered bootstrap analysis, and manuscript figures.

## Study design

The analysis includes:

- Primary controlled benchmarks: MovieLens 100K and MovieLens 1M
- Multi-domain replication datasets: Book-Crossing, FilmTrust, and Jester
- User-stratified approximately 80/10/10 train/validation/test splits
- Fixed validation and test sets within each split
- Nested training-retention levels of 100%, 75%, 50%, 25%, and 10%
- Minimum of 5 retained training ratings per user
- Four controlled models:
  - BiasOnly
  - BiasMF
  - CosineMF
  - NormalizedMF
- Evaluation metrics: RMSE and MAE
- MovieLens 100K: 10 initialization seeds
- MovieLens 1M: 5 initialization seeds
- Book-Crossing: 5 initialization seeds
- FilmTrust: 5 initialization seeds
- Jester: 3 initialization seeds
- Sensitivity analysis over latent dimension and weight decay
- Alternative split-seed robustness analysis
- Warm-item-only robustness analysis
- User-clustered paired bootstrap for the primary MovieLens comparison
- Jester native-rating-scale sensitivity analysis

Repeated initialization seeds quantify **optimization variability**. They are not treated as independent population samples.

The user-clustered bootstrap evaluates variation in the composition of test users conditional on the fitted models and benchmark datasets.

---

## Core model comparison

Let $r_{ui}$ denote the observed rating assigned by user $u$ to item $i$, and let $\hat{r}_{ui}$ denote the corresponding prediction.

### BiasOnly

The bias-only model contains the global mean rating and user- and item-specific bias terms:

```math
\hat{r}_{ui} = \mu + b_u + b_i
```

This model provides a low-complexity reference for assessing whether latent interactions remain useful as rating information becomes sparse.

### BiasMF

The standard biased matrix-factorization model is:

```math
\hat{r}_{ui} = \mu + b_u + b_i + p_u^\top q_i
```

where $p_u$ and $q_i$ are $K$-dimensional user and item latent vectors.

The latent interaction can be decomposed as:

```math
p_u^\top q_i = \lVert p_u \rVert_2 \lVert q_i \rVert_2 \cos(\theta_{ui})
```

This decomposition shows that conventional matrix factorization depends jointly on latent-vector direction and magnitude.

### L2 normalization

For the normalized models, user and item latent vectors are transformed to unit norm:

```math
\tilde{p}_u = \frac{p_u}{\lVert p_u \rVert_2}, \qquad
\tilde{q}_i = \frac{q_i}{\lVert q_i \rVert_2}
```

This removes user- and item-specific radial magnitude from the latent interaction.

### CosineMF

CosineMF applies one learned positive global scale $s$ to the normalized latent interaction:

```math
\hat{r}_{ui}
=
\mu + b_u + b_i
+
s\tilde{p}_u^\top\tilde{q}_i
```

Equivalently:

```math
\hat{r}_{ui}
=
\mu + b_u + b_i
+
s\sum_{k=1}^{K}\tilde{p}_{uk}\tilde{q}_{ik}
```

All normalized latent coordinates therefore share the same interaction scale $s$.

### NormalizedMF

NormalizedMF replaces the single global scale with $K$ positive dimension-specific scales:

```math
\hat{r}_{ui}
=
\mu + b_u + b_i
+
\sum_{k=1}^{K}s_k\tilde{p}_{uk}\tilde{q}_{ik}
```

Equivalently:

```math
\hat{r}_{ui}
=
\mu + b_u + b_i
+
\tilde{p}_u^\top D_s\tilde{q}_i
```

where:

```math
D_s = \mathrm{diag}(s_1,\ldots,s_K)
```

Each scale is constrained to be positive using:

```math
s_k = \mathrm{softplus}(\alpha_k)
```

The central controlled comparison is therefore:

- **CosineMF:** one positive global scale $s$
- **NormalizedMF:** $K$ positive dimension-specific scales $s_1,\ldots,s_K$

The two models use the same L2-normalized latent factors and differ only in how interaction strength is parameterized after normalization.

The learned $s_k$ values should not be interpreted as stable semantic feature-importance measures because latent coordinates are not semantically identified.

---

## Primary paired comparison

The direct comparison between CosineMF and NormalizedMF is defined as:

```math
\Delta \mathrm{RMSE}
=
\mathrm{RMSE}(\mathrm{CosineMF})
-
\mathrm{RMSE}(\mathrm{NormalizedMF})
```

Positive values indicate lower RMSE for NormalizedMF.

The relative RMSE reduction is:

```math
\frac{
\mathrm{RMSE}(\mathrm{CosineMF})
-
\mathrm{RMSE}(\mathrm{NormalizedMF})
}{
\mathrm{RMSE}(\mathrm{CosineMF})
}
\times 100
```

Across the harmonized 1–5 rating-scale analyses, NormalizedMF had lower RMSE than CosineMF in 24 of 25 dataset-retention conditions.

The only negative condition was MovieLens 1M at 75% nominal retention, where the difference was very small.

These counts summarize directional consistency on the evaluated benchmark datasets and should not be interpreted as evidence of universal superiority.

---

## Data and sparsification

Only the training set is sparsified.

For each user, one random ordering of the training ratings is fixed. Lower-retention datasets are constructed as nested prefixes of that same ordering.

The nominal training-retention levels are:

- 100%
- 75%
- 50%
- 25%
- 10%

At least 5 training observations per user are retained to avoid artificially creating cold users.

Because of this minimum-history constraint, nominal and realized retention levels may differ, particularly in datasets with many low-activity users.

At the nominal 10% condition, realized retention is approximately:

| Dataset | Realized retention |
|---|---:|
| MovieLens 100K | 11.24% |
| MovieLens 1M | 10.55% |
| Book-Crossing | 35.35% |
| FilmTrust | 20.94% |
| Jester | 10.78% |

Cross-dataset comparisons should therefore be interpreted using both nominal and realized retention.

---

## Rating-scale harmonization

The three replication datasets are linearly mapped to a common 1–5 scale before model fitting:

- Book-Crossing: 1–10 to 1–5
- FilmTrust: 0.5–4 to 1–5
- Jester: -10–10 to 1–5

This harmonization keeps squared-error metrics and shared optimizer calibration more comparable across datasets.

A supplementary Jester analysis is also performed on its native -10 to 10 scale while retaining the same optimizer settings.

Under that unmatched calibration, the direct CosineMF–NormalizedMF comparison reverses direction.

This sensitivity result is retained intentionally because it shows that the relative performance of the two parameterizations is not invariant to rating scale and optimizer calibration.

---

## Main experimental settings

The main fixed settings are:

| Setting | Value |
|---|---:|
| Latent dimension | 32 |
| Adam learning rate | 0.015 |
| Weight decay | 1e-5 |
| Minimum retained ratings per user | 5 |
| Primary split seed | 20260918 |

### MovieLens 100K

- Batch size: 16,384
- Maximum epochs: 12
- Early-stopping patience: 3
- Initialization seeds: 10

### MovieLens 1M

- Batch size: 131,072
- Maximum epochs: 8
- Early-stopping patience: 2
- Initialization seeds: 5

### Book-Crossing

- Batch size: 16,384
- Maximum epochs: 15
- Early-stopping patience: 3
- Initialization seeds: 5

### FilmTrust

- Batch size: 16,384
- Maximum epochs: 15
- Early-stopping patience: 3
- Initialization seeds: 5

### Jester

- Batch size: 131,072
- Maximum epochs: 10
- Early-stopping patience: 2
- Initialization seeds: 3

The sensitivity analysis additionally evaluates:

```math
K \in \{16,32,64\}
```

and weight-decay values:

```math
\lambda \in \{10^{-5},10^{-4},10^{-3}\}
```

These shared settings are used to isolate the scale-parameterization comparison. They should not be interpreted as independently optimized hyperparameters for every model.

---

## Evaluation metrics

Prediction accuracy is evaluated using RMSE and MAE.

### Root mean squared error

For $N$ test observations:

```math
\mathrm{RMSE}
=
\sqrt{
\frac{1}{N}
\sum_{(u,i)}
(r_{ui}-\hat{r}_{ui})^2
}
```

### Mean absolute error

```math
\mathrm{MAE}
=
\frac{1}{N}
\sum_{(u,i)}
\left|r_{ui}-\hat{r}_{ui}\right|
```

Lower RMSE and MAE indicate better rating-prediction accuracy.

Because the manuscript focuses on **explicit rating prediction** rather than ranked recommendation, ranking metrics such as NDCG and Recall@K are not used.

---

## User-clustered paired bootstrap

For the primary MovieLens benchmarks, uncertainty in the paired RMSE difference is evaluated using a user-clustered bootstrap.

Users in the fixed test set are sampled with replacement 10,000 times.

All test ratings belonging to a sampled user are kept together.

For each bootstrap resample:

1. RMSE is recomputed for CosineMF.
2. RMSE is recomputed for NormalizedMF.
3. The paired RMSE difference is calculated for each matched initialization seed.
4. The paired differences are averaged across the fixed seed set.

The resulting percentile interval describes **test-user sampling variation conditional on the evaluated benchmark datasets and fitted models**.

It is not a population-level confidence interval and does not establish generalization to unseen datasets.

The bootstrap results are stored in:

`results/user_cluster_bootstrap_ci.csv`

---

## Robustness analyses

### Latent dimension and weight decay

The direct CosineMF–NormalizedMF comparison is repeated on MovieLens 100K across:

```math
K \in \{16,32,64\}
```

and:

```math
\lambda \in \{10^{-5},10^{-4},10^{-3}\}
```

The analysis is conducted at 100%, 50%, and 10% nominal retention.

### Alternative train/validation/test splits

The direct comparison is repeated using three independent data-split seeds on MovieLens 100K at:

- 100% retention
- 50% retention
- 10% retention

### Warm-item-only analysis

Training-set sparsification can cause some test items to become unseen in a sparse training subset.

The 10% retention analysis is therefore repeated after removing those test observations.

This analysis evaluates whether sparse-subset cold items explain the main CosineMF–NormalizedMF difference.

---

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

See [`ENVIRONMENT.md`](ENVIRONMENT.md) for additional notes on reproducibility across platforms.

---

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

### Git Bash on Windows

```bash
source .venv/Scripts/activate
```

### Linux or macOS

```bash
source .venv/bin/activate
```

Upgrade pip and install the required packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Data

Raw third-party datasets are **not redistributed** in this repository.

Place the required source files under the `data/` directory before running the analyses.

See [`data/README.md`](data/README.md) for the expected filenames and preprocessing instructions.

The public datasets used in the manuscript are:

- MovieLens 100K
- MovieLens 1M
- Book-Crossing
- FilmTrust
- Jester

Users should obtain these datasets from their respective original public sources and comply with their licenses and terms of use.

---

## Recommended reproduction order

### 1. Primary MovieLens experiments

```bash
python scripts/01_main_experiments.py --dataset both
```

### 2. Latent-dimension and regularization sensitivity

```bash
python scripts/02_sensitivity_analysis.py
```

### 3. Warm-item robustness analysis

```bash
python scripts/03_warm_item_robustness.py
```

### 4. Alternative split-seed robustness

```bash
python scripts/04_split_robustness.py
```

### 5. Multi-domain replication

```bash
python scripts/07_external_validation.py --dataset all
```

### 6. Jester native-scale sensitivity analysis

```bash
python scripts/07_external_validation.py --dataset Jester --jester-native-scale
```

### 7. Manuscript-level summaries

```bash
python scripts/05_make_summaries.py
```

### 8. User-clustered paired bootstrap

```bash
python scripts/08_user_cluster_bootstrap.py --dataset both --bootstrap-reps 10000
```

### 9. Publication figures

```bash
python scripts/06_make_final_figures.py
```

The full experiment set can be computationally expensive, particularly MovieLens 1M, Jester, and the 10,000-resample bootstrap analysis.

---

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
├── src/
│   └── lfn/
│       ├── __init__.py
│       └── core.py
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
│   ├── ML100K_validated_summary.csv
│   ├── ML1M_validated_summary.csv
│   ├── ML100K_cosine_norm_sensitivity_summary.csv
│   ├── ML100K_split_robustness.csv
│   ├── warm_item_10pct_robustness_summary.csv
│   ├── user_cluster_bootstrap_ci.csv
│   ├── Jester_native_scale_ablation.csv
│   ├── five_dataset_direct_ablation.csv
│   ├── retention_metadata.csv
│   └── external_validation/
├── figures/
│   └── README.md
├── supplementary/
│   ├── Supplementary_Material.docx
│   ├── Supplementary_tables.zip
│   └── README.md
└── .github/
    └── workflows/
        └── python-smoke.yml
```

---

## Manuscript-level processed outputs

The repository includes processed results supporting the manuscript and supplementary material.

Key files include:

- `results/ML100K_validated_summary.csv`
- `results/ML1M_validated_summary.csv`
- `results/ML100K_cosine_norm_sensitivity_summary.csv`
- `results/ML100K_split_robustness.csv`
- `results/warm_item_10pct_robustness_summary.csv`
- `results/user_cluster_bootstrap_ci.csv`
- `results/Jester_native_scale_ablation.csv`
- `results/five_dataset_direct_ablation.csv`
- `results/retention_metadata.csv`
- external-validation summaries under `results/external_validation/`

Long per-seed outputs can be regenerated using the analysis scripts and are not required to remain under version control.

---

## Main manuscript figures

The figure-generation pipeline produces:

- **Figure 1:** Test RMSE under nested controlled sparsity on MovieLens
- **Figure 2:** Cross-dataset direct CosineMF–NormalizedMF ablation
- **Figure 3:** Sensitivity of the MovieLens 100K direct normalization ablation

The nominal-versus-realized retention comparison is reported as **Supplementary Fig. S1**.

---

## Main empirical findings

Under the harmonized 1–5 protocol:

- NormalizedMF had lower RMSE than CosineMF at all five MovieLens 100K retention levels.
- NormalizedMF had lower RMSE at four of five MovieLens 1M retention levels.
- MovieLens 1M at 75% retention showed a small reversal.
- The user-clustered bootstrap interval was positive in 9 of the 10 primary MovieLens conditions.
- NormalizedMF had lower RMSE than CosineMF in all 15 Book-Crossing, FilmTrust, and Jester replication conditions.
- Across all five datasets, the direct comparison favored NormalizedMF in 24 of 25 dataset-retention conditions.

These findings concern the **direct parameterization comparison between CosineMF and NormalizedMF**.

They do not imply that latent-factor models are always the best overall predictors.

BiasOnly achieved the lowest RMSE in several sparse settings, including the lowest-retention MovieLens conditions and all Book-Crossing retention levels.

---

## Interpretation

The central question of the study is:

> After user- and item-specific latent-vector magnitude has been removed through L2 normalization, does allowing separate positive scales for individual latent dimensions provide a useful refinement over using one global interaction scale?

CosineMF imposes one scale $s$ across all normalized latent coordinates.

NormalizedMF instead allows $s_1,\ldots,s_K$ to vary across latent coordinates.

This adds coordinate-specific flexibility after normalization while introducing only $K$ additional scale parameters.

Prediction complexity remains $O(K)$.

The study does **not** claim that normalization solves missing-data or cold-start problems.

The findings should instead be interpreted as evidence about the parameterization of a normalized latent interaction.

---

## Important limitations

1. The two primary controlled benchmarks are both MovieLens datasets.
2. The external datasets differ in realized retention because the minimum-five-training-rating rule affects datasets differently.
3. Shared optimizer settings are used to control the direct comparison, but these settings are not guaranteed to be individually optimal for every model.
4. The native-scale Jester analysis reverses the direct comparison when optimizer settings are held fixed, demonstrating sensitivity to rating-scale calibration.
5. The experiments manipulate the amount of observed rating information among users with interaction histories. They do not constitute a direct solution to new-user or new-item cold start.
6. Initialization-seed variability and user-clustered bootstrap variability represent different sources of uncertainty.
7. The bootstrap intervals are conditional on the evaluated datasets and fitted models and are not population-level confidence intervals.

---

## Reproducibility note

Fixed random seeds substantially reduce stochastic variation, but bit-for-bit identical floating-point output is not guaranteed across:

- hardware,
- operating systems,
- BLAS implementations,
- PyTorch versions and builds,
- and other numerical-library configurations.

Reproduction should therefore focus on manuscript-level means, paired model differences, and qualitative patterns rather than exact agreement in the final decimal place.

---

## License

The source code in this repository is released under the [MIT License](LICENSE).

Third-party datasets are not included in this repository and remain subject to their respective licenses and terms of use.

---

## Citation

If you use this repository, please cite the associated manuscript.

Citation metadata are provided in [`CITATION.cff`](CITATION.cff).
