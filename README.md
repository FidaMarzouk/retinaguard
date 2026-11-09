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

Models are trained on EyePACS and evaluated per site on external datasets (India, China),
with bootstrap 95% confidence intervals.

## Results

Referable diabetic retinopathy (ICDR grade ≥ 2), one image per eye, AUROC with patient-clustered
bootstrap 95% CIs. Trained on EyePACS only; the external sites were never used for training or
model selection.

| Test set | Images (referable) | ConvNeXt-Tiny | ConvNeXt-Tiny + LMN | EfficientNet-B3 + LMN |
|---|---|---|---|---|
| EyePACS internal test (US) | 3,252 (617) | 0.950 (0.938–0.961) | 0.951 (0.939–0.961) | 0.948 (0.935–0.959) |
| APTOS 2019 (India) | 3,517 (1,379) | 0.971 (0.967–0.976) | 0.972 (0.967–0.977) | 0.965 (0.959–0.970) |
| IDRiD (India) | 508 (318) | **0.988** (0.981–0.994) | 0.982 (0.972–0.991) | 0.983 (0.973–0.991) |
| DDR (China) | 12,354 (5,604) | 0.973 (0.970–0.975) | 0.971 (0.969–0.974) | 0.962 (0.959–0.966) |

LMN is local mean normalisation (subtracting a blurred copy of the image to even out
illumination). The production model is ConvNeXt-Tiny on plain images: it was chosen on
validation AUROC (0.959 against 0.953 with LMN), and on test it is never significantly worse
(paired bootstrap), significantly better on IDRiD, and has a higher quadratic kappa on every
site. With LMN, ConvNeXt-Tiny is significantly better than EfficientNet-B3 on APTOS and DDR. External AUROC is
higher than internal mainly because of case mix: on the hard boundary (grade 1 vs grade 2) it is
0.82–0.94 everywhere. The main failure mode is end-stage disease (vitreous haemorrhage,
fibrosis) predicted as no DR: 83 of 906 DDR grade 4 images (101 with LMN).

### Triage

Scores are temperature-scaled on the validation split. The clearing threshold is set on a
separate calibration split by conformal risk control, so that on average at most 5% of
referable eyes are cleared as NO_REFER; REFER needs a score with at least 80% precision on that
split, and everything in between is deferred to a human.

| Test set | Referable eyes cleared (FNR) | NO_REFER | REFER | Deferred |
|---|---|---|---|---|
| EyePACS internal test | 2.8% (1.4–4.2) | 49.3% | 19.7% | 31.0% |
| APTOS 2019 | 0.0% | 31.0% | 49.4% | 19.6% |
| IDRiD | 0.0% | 17.3% | 66.1% | 16.5% |
| DDR | 1.6% (1.3–1.9) | 36.5% | 43.1% | 20.4% |

The guarantee only covers images like the calibration ones; the external rates are measured, not
guaranteed. The calibration fitted on EyePACS does not transfer to APTOS, whose grading runs a
step higher: its referable ECE stays at 0.10 after scaling (0.02 on EyePACS).

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

Training is configured with Hydra (`configs/`) and can also run locally:

```bash
uv run python -m retinaguard.train.train_grader model=efficientnet_b3 "data.shard_dirs=[path/to/shards]"
```

A run directory keeps `last.pt` after every epoch; rerunning the same command continues where a
stopped run left off.

Run folders downloaded from Kaggle are imported into MLflow (parameters, per-epoch metrics, test
metrics from `evaluate`); `--register` makes that run's `best.pt` the `champion` version of the
`dr-grader` model:

```bash
uv run python -m retinaguard.train.import_runs outputs/runs/grader-convnext_tiny \
    --predictions-root data/interim/predictions --register grader-convnext_tiny
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db   # or set MLFLOW_TRACKING_URI
```

Temperature scaling and the deferral thresholds are fitted from a model's predictions; the
targets are in `configs/policy/decision.yaml`, and the output `policy.json` feeds
`retinaguard.serving.policy`:

```bash
uv run python -m retinaguard.train.calibrate --manifest data/manifests/manifest.parquet \
    --predictions data/interim/predictions/grader-convnext_tiny-no_lmn/predictions.parquet \
    --out-dir data/interim/predictions/grader-convnext_tiny-no_lmn/policy
```

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
| IDRiD (Porwal et al., IEEE DataPort, [doi:10.21227/H25W98](https://doi.org/10.21227/H25W98)) | Lesion segmentation, external test (India) | CC BY 4.0 |
| DDR (Li et al., Information Sciences 2019, [doi:10.1016/j.ins.2019.06.011](https://doi.org/10.1016/j.ins.2019.06.011)) | Lesion segmentation, external test (China) | CC BY-NC-SA 4.0 |

Only IDRiD images (CC BY 4.0, attributed) may appear in tests or the public demo.
`tests/fixtures/idrid/` holds four IDRiD disease grading training images (IDRiD_005, 016, 138,
139), downscaled from 4288×2848 to 1072×712 and re-encoded as JPEG.
