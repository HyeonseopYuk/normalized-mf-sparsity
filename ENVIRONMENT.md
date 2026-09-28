# Reproducibility environment

The finalized manuscript analysis was run with the following environment:

| Component | Version used |
|---|---|
| Repository release | 1.4.0 |
| Python | 3.13.5 |
| NumPy | 2.3.5 |
| pandas | 2.2.3 |
| SciPy | 1.17.0 |
| PyTorch | 2.10.0+cpu |
| Matplotlib | 3.10.8 |
| openpyxl | 3.1.5 |
| python-docx | 1.2.0 |

`requirements.txt` pins `torch==2.10.0` because the local `+cpu` build tag is distribution-channel specific. The finalized manuscript calculations used the CPU build `2.10.0+cpu`.

Small floating-point differences can occur across operating systems, BLAS backends, CPU/GPU devices, and PyTorch builds even with fixed seeds. Repeated initialization seeds in the manuscript quantify optimization stability rather than independent population sampling.
