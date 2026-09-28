# Figures

Publication figures are regenerated with:

```bash
python scripts/06_make_final_figures.py
```

Expected outputs include:

- `Figure1_cross_dataset_RMSE.*`
- `Figure2_direct_ablation_heatmap.*`
- `Figure3_sensitivity_heatmaps.*`
- `Supplementary_Figure_S1_realized_retention_deviation.*`

Generated PNG/PDF/TIFF files are ignored by Git because they can be recreated from the processed result tables.
