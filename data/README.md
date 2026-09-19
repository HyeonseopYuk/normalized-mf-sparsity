# Dataset Setup

Raw datasets are not redistributed in this repository. Please download them from their original sources and place them in the paths below.

## MovieLens 100K
Source: GroupLens Research

```text
data/ml-100k/u.data
```

## MovieLens 1M
Source: GroupLens Research

```text
data/ml-1m/ratings.dat
```

## Book-Crossing
Source: Book-Crossing public dataset

Use the explicit-rating data required by the preprocessing script. Ratings of 0 are excluded.

## FilmTrust
Source: FilmTrust public dataset

```text
data/filmtrust/ratings.txt
```

## Jester
Source: Jester Joke Recommender System

```text
data/jester/jester-data-1.csv
```

The value `99` is treated as missing.

## Notes

The repository code performs all preprocessing, including:
- user-stratified train/validation/test splitting;
- nested retention at 100%, 75%, 50%, 25%, and 10%;
- the minimum-five-training-rating rule;
- rating-scale harmonization for Book-Crossing, FilmTrust, and Jester.

See the main `README.md` for the execution order and software requirements.
