# Results

This directory contains manuscript-level processed summaries and is also used for regenerated analysis outputs.

Positive direct-ablation values are defined as:

`RMSE(CosineMF) - RMSE(NormalizedMF)`

so positive values favor the dimension-specific NormalizedMF parameterization.

Key files include:

- `ML100K_validated_summary.csv`
- `ML1M_validated_summary.csv`
- `ML100K_cosine_norm_sensitivity_summary.csv`
- `five_dataset_direct_ablation.csv`
- `retention_metadata.csv`
- `user_cluster_bootstrap_ci.csv`
- `ML100K_split_robustness.csv`
- `warm_item_10pct_robustness_summary.csv`
- `Jester_native_scale_ablation.csv`
- `external_validation/*_summary_clean.csv`
- `external_validation/*_ablation_clean.csv`

The bootstrap intervals resample test users while holding the trained seed pairs fixed. They quantify test-user sampling variation conditional on the evaluated benchmark design, not population-level uncertainty.

Long per-seed outputs can be regenerated from raw data and are ignored by default in `.gitignore`.
