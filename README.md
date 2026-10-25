# RetinaGuard

> **Disclaimer:** RetinaGuard is a research and portfolio prototype. It is **not a medical device** and must not be used for clinical decisions.

Quality-aware, uncertainty-aware diabetic retinopathy (DR) screening triage:
quality gate → calibrated grader → conformal deferral → lesion overlays → drift monitoring.

For each fundus image RetinaGuard returns `RETAKE`, `REFER`, `NO_REFER` or `DEFER_TO_HUMAN`. A conformal-risk-control threshold bounds the false-negative rate of `NO_REFER`, and unfamiliar images are deferred to a human.

## Status

Phase 0 (project setup). See [PLAN.md](PLAN.md) for the full plan and phase checklist.

## Results

_To be filled from [docs/results.md](docs/results.md)._

## Architecture

_See [PLAN.md §3](PLAN.md#3-architecture); `docs/architecture.md` follows in Phase 8._

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/), GNU make, Docker (for the stack).

```bash
make setup     # install Python 3.11 env + pre-commit hooks
make lint      # ruff + mypy
make test      # pytest
make up        # Postgres, MLflow (http://localhost:5000), API (http://localhost:8000/docs), app (http://localhost:8501)
make down
```

Local alternatives: `make serve` (API with reload), `make app` (Streamlit), `make pipeline` (Dagster assets).

## Data and licences

No images, datasets or model weights are stored in this repository. Datasets are obtained separately under their own terms (see [PLAN.md §5](PLAN.md#5-datasets)); `data/` is git-ignored and versioned with DVC. Only IDRiD images (CC BY 4.0, attributed) appear in tests or the public demo.
