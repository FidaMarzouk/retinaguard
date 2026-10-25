# 0004: Large datasets stay on Kaggle

- Status: accepted
- Date: 2026-10-25

## Context
EyePACS (~88 GB), APTOS (~10 GB) and DDR (~13 GB) exceed the local disk budget (~40 GB free), and
training runs on Kaggle GPUs anyway.

## Decision
- EyePACS and APTOS are read in Kaggle notebooks from the attached competition data; DDR is pulled
  into a Kaggle notebook from its Hugging Face mirror. None are downloaded locally.
- Kaggle notebooks run the same `retinaguard` code (installed from GitHub) to produce 512 px images
  and manifest rows, saved as **private** Kaggle datasets.
- Locally (`data/raw/`): label files, IDRiD, Messidor-2 (its licence forbids re-uploading) and
  EyeQ labels, roughly 3–4 GB.
- The combined manifest and splits are built locally from the per-source manifests, so the §7.2
  invariants are checked over all sources in one place.

## Consequences
Local Dagster assets operate on manifests, not on the EyePACS/APTOS/DDR pixels; image-level checks
for those sources (hashes, preprocessing failures) are computed on Kaggle and travel in the
manifest.
