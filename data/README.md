# Data files

Raw third-party datasets are not redistributed in this repository.

Place the following files in this directory before running the corresponding analyses:

```text
data/
├── ml-100k.zip
├── ratings.dat                         # MovieLens 1M ratings file
│   # alternatively: ml-1m.zip
├── BX-Book-Explicit-5Rate-Map.csv      # Book-Crossing explicit-rating file used in this study
├── ratings.txt                         # FilmTrust ratings
└── jester-data-1.csv                   # Jester Dataset 1 wide matrix
```

## Preprocessing used in the manuscript

- **MovieLens 100K / 1M**: user-stratified approximately 80/10/10 train/validation/test split.
- **Book-Crossing**: retain users with at least 7 observed ratings, then linearly map ratings from 1-10 to 1-5.
- **FilmTrust**: retain users with at least 7 observed ratings, then linearly map ratings from 0.5-4 to 1-5.
- **Jester**: treat 99 as missing; for the harmonized analysis, map -10..10 to 1..5 using `r_scaled = 3 + 0.2*r`.
- **Jester native-scale sensitivity**: run with `--jester-native-scale` to retain the original -10..10 scale.

Only the training split is sparsified. Validation and test data remain fixed. At least five training observations per user are retained, so nominal and realized retention can differ, particularly for Book-Crossing and FilmTrust.

Please obtain each dataset from its original public research release and comply with its license or terms of use.
