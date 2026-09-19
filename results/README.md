# Results

This directory is populated by the analysis scripts.

`external_validation/` contains the manuscript-level summary and direct-ablation tables for Book-Crossing, FilmTrust, and Jester that were used for cross-domain validation. Long per-seed outputs can be regenerated from raw data with `scripts/07_external_validation.py`.

Positive direct-ablation values are defined as:

`RMSE(CosineMF) - RMSE(NormalizedMF)`

so positive values favor the dimension-scaled NormalizedMF parameterization.
