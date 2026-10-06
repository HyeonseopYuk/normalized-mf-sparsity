# Resubmission robustness extension

This directory is populated by `scripts/09_fair_tuning_ncf.py`.

## Purpose

The extension addresses three reviewer-facing concerns without changing the primary
scientific question of the paper:

1. **Model-specific hyperparameter tuning** using the validation set only.
2. **A stronger neural baseline** (a NeuMF-style NCF regressor for explicit ratings).
3. **Non-parametric paired inference** at the user level with Wilcoxon signed-rank
   tests and Holm correction.

## Scope

To keep the additional experiment focused and computationally tractable, it is run
on the two primary MovieLens datasets at training-retention levels 100%, 50%, and
10%. The train/validation/test partition and nested sparsification logic are exactly
the same as in the main study.

For each model and retention level, the validation grid is:

- latent dimension: 16, 32
- learning rate: 0.005, 0.015
- weight decay: 1e-5, 1e-4

The selected setting is the one with the lowest validation RMSE using tuning seed
707. The selected configuration is then evaluated with seeds 11, 22, 33, 44, and
55. Test data are not used for hyperparameter selection.

## Models

- BiasMF
- CosineMF
- NormalizedMF
- NCF (NeuMF-style explicit-rating regressor with user/item biases)

The NCF baseline is included as a contemporary neural comparator, not as a claim
that it represents the current state of the art across all recommendation settings.

## Statistical comparison

For each retention level, predictions are averaged over evaluation seeds. User-level
RMSE is then computed on the common fixed test set. NormalizedMF is compared with
CosineMF, BiasMF, and NCF using paired two-sided Wilcoxon signed-rank tests. Holm
adjustment is applied across the three comparisons within each dataset-retention
condition.

## Run

```bash
python scripts/09_fair_tuning_ncf.py --dataset both
```

Required raw public data files are described in `data/README.md`; they are not
redistributed in this repository.

## Outputs

- `ML100K_model_specific_tuning.csv`
- `ML100K_tuned_evaluation_long.csv`
- `ML100K_tuned_evaluation_summary.csv`
- `ML100K_user_level_wilcoxon.csv`
- `ML100K_selected_hyperparameters.json`
- corresponding `ML1M_...` files

These results should be incorporated into the manuscript only after the script has
completed successfully and the outputs have been checked against the fixed-split
protocol.
