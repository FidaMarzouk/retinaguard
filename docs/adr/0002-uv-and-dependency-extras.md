# 0002: uv with per-component dependency extras

- Status: accepted
- Date: 2026-10-23

## Context
The project mixes very different runtimes: a CPU inference API, a Streamlit app, a Dagster pipeline, and GPU training on Kaggle. Installing everything everywhere makes Docker images large and CI slow.

## Decision
- uv manages the environment and lockfile (`uv.lock`), with Python pinned to 3.11.
- Runtime dependencies are split into extras: `api`, `app`, `pipeline`. Each Docker image installs only its extra.
- Dev tooling (ruff, mypy, pytest, pre-commit, DVC) goes in the `dev` dependency group.
- Training dependencies (torch, timm, albumentations, segmentation-models-pytorch, ONNX, MLflow client) will be added in Phase 2 as an `ml` extra. Torch's CPU and CUDA wheels need index configuration, which is decided once there is training code to test it against.

## Consequences
`make setup` installs all extras locally. CI does the same, so imports from every component are type-checked and tested.
