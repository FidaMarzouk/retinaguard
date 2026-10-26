# RetinaGuard

Quality-aware, uncertainty-aware triage for diabetic retinopathy (DR) screening on fundus photographs.


## What it does

For each fundus image RetinaGuard returns one of four decisions:

| Decision | When |
|---|---|
| `RETAKE` | The quality model rejects the image |
| `DEFER_TO_HUMAN` | The image looks out of distribution, or the model is uncertain |
| `NO_REFER` | The calibrated probability of referable DR is below a conformal threshold that bounds the false-negative rate |
| `REFER` | The probability of referable DR is high |

Pipeline: image quality gate → calibrated DR grader → out-of-distribution check → conformal
deferral policy → lesion overlays → drift monitoring. Any error in the chain returns
`DEFER_TO_HUMAN`, never `NO_REFER`.

Models are trained on EyePACS and evaluated per site on external datasets (India, France, China),
with bootstrap 95% confidence intervals.

## Results

_In progress._

## Quick start

Requirements: Python 3.11, [uv](https://docs.astral.sh/uv/), GNU make, Docker.

```bash
make setup     # create the environment and install pre-commit hooks
make lint      # ruff + mypy
make test      # pytest
make up        # Postgres, MLflow (http://localhost:5000), API (http://localhost:8000/docs), app (http://localhost:8501)
make down
```

Heavy preprocessing and training run as Kaggle notebooks (`notebooks/kaggle/`), which install
this package from GitHub.

## Repository layout

```
src/retinaguard/
  data/        dataset adapters, manifest schema, preprocessing, hashing
  models/      grader, quality model, calibration, conformal deferral, OOD
  train/       training and evaluation
  pipeline/    Dagster assets and data checks
  serving/     FastAPI inference service
  app/         Streamlit dashboard
  monitoring/  drift reports
notebooks/kaggle/   GPU/CPU jobs run on Kaggle
tests/              unit, integration and regression tests
```

## Data and licences

No images, datasets or model weights are stored in this repository; `data/` is git-ignored and
versioned with DVC. Each dataset is obtained separately under its own terms:

| Dataset | Role | Licence / access |
|---|---|---|
| EyePACS (Kaggle Diabetic Retinopathy Detection, 2015) | Training, internal test | Kaggle competition rules |
| EyeQ | Image quality labels | Labels on GitHub (HzFu/EyeQ); images from EyePACS |
| APTOS 2019 | External test (India) | Kaggle competition rules |
| Messidor-2 | External test (France) | ADCIS research licence |
| IDRiD | Lesion segmentation, external test (India) | CC BY 4.0 |
| DDR | Lesion segmentation, external test (China) | CC BY 4.0 (as published by the authors) |

Only IDRiD images (CC BY 4.0, attributed) may appear in tests or the public demo.
