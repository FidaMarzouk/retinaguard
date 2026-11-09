"""Calibrate a grader's predictions and fit the deferral policy, then report both per site.

    python -m retinaguard.train.calibrate --manifest manifest.parquet \\
        --predictions predictions/grader-convnext_tiny/predictions.parquet \\
        --out-dir predictions/grader-convnext_tiny/policy

Temperatures of both heads are fitted on ``val``, the policy thresholds on ``calib`` (with the
calibrated scores); both EyePACS splits leave out EyeQ rejects, which the quality gate would
send to RETAKE. Test sets are only measured. Writes:

- ``policy.json``: temperatures and thresholds (loadable as ``serving.policy.Thresholds``);
- ``predictions.parquet``: the calibrated predictions;
- ``calibration.csv``: ECE and Brier score per set, before and after scaling;
- ``deferral.csv``: per set, the share of each decision and the realised false-negative rate
  (referable eyes cleared), with 95% bootstrap CIs over ``split_group`` clusters;
- ``reliability.csv`` and ``coverage.csv``: reliability bins, and accuracy against coverage
  when the least confident images are deferred.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from omegaconf import OmegaConf

from retinaguard.models.calibration import (
    brier_score,
    expected_calibration_error,
    fit_binary_temperature,
    fit_class_temperature,
    multiclass_brier_score,
    reliability_table,
    scale_binary,
    scale_classes,
)
from retinaguard.models.conformal import lambda_high, lambda_low
from retinaguard.models.metrics import bootstrap_ci
from retinaguard.serving.policy import Thresholds
from retinaguard.train.evaluate import evaluation_sets
from retinaguard.train.predict import PROBABILITY_COLUMNS

DEFAULT_POLICY = Path("configs/policy/decision.yaml")
COVERAGES = np.round(np.arange(0.5, 1.0001, 0.05), 2)


def fitting_rows(manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    """Graded, processed EyePACS images of ``split`` that pass the quality gate."""
    rows = manifest[
        (manifest["split"] == split)
        & manifest["dr_grade"].notna()
        & manifest["processed_path"].notna()
    ]
    return rows[rows["quality_label"].ne("reject").fillna(True)]


def calibrate_predictions(
    predictions: pd.DataFrame, temperature_referable: float, temperature_grade: float
) -> pd.DataFrame:
    calibrated = predictions.copy()
    calibrated[PROBABILITY_COLUMNS] = scale_classes(
        predictions[PROBABILITY_COLUMNS].to_numpy(), temperature_grade
    )
    calibrated["p_referable"] = scale_binary(predictions["p_referable"], temperature_referable)
    return calibrated


def _join(rows: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    joined = rows.merge(predictions, on="image_id", how="left", validate="one_to_one")
    missing = joined["p_referable"].isna()
    if missing.any():
        raise ValueError(f"{int(missing.sum())} images have no prediction")
    return joined


def _top_label(joined: pd.DataFrame) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
    probs = joined[PROBABILITY_COLUMNS].to_numpy()
    correct = probs.argmax(1) == joined["dr_grade"].astype(int).to_numpy()
    return correct, probs.max(1)


def calibration_metrics(joined: pd.DataFrame) -> dict[str, float]:
    referable = (joined["dr_grade"].astype(int) >= 2).to_numpy()
    correct, confidence = _top_label(joined)
    return {
        "referable_ece": expected_calibration_error(referable, joined["p_referable"]),
        "referable_brier": brier_score(referable, joined["p_referable"]),
        "grade_ece": expected_calibration_error(correct, confidence),
        "grade_brier": multiclass_brier_score(
            joined["dr_grade"].astype(int), joined[PROBABILITY_COLUMNS]
        ),
    }


def deferral_metrics(
    joined: pd.DataFrame, thresholds: Thresholds, n_resamples: int = 2000, seed: int = 0
) -> dict[str, float | int]:
    """Decision shares and realised false-negative rate, with cluster bootstrap CIs."""
    referable = (joined["dr_grade"].astype(int) >= 2).to_numpy()
    p = joined["p_referable"].to_numpy()
    cleared = p < thresholds.lambda_low
    referred = p >= thresholds.lambda_high

    def fnr(truth: npt.NDArray[np.generic], score: npt.NDArray[np.generic]) -> float:
        positives = truth.astype(bool)
        if not positives.any():
            return float("nan")
        return float(np.mean(score[positives] < thresholds.lambda_low))

    def share(flags: npt.NDArray[np.generic], _: npt.NDArray[np.generic]) -> float:
        return float(np.mean(flags.astype(bool)))

    result: dict[str, float | int] = {"images": len(joined), "referable": int(referable.sum())}
    estimates = {
        "fnr": (fnr, referable, p),
        "no_refer_rate": (share, cleared, p),
        "refer_rate": (share, referred, p),
        "defer_rate": (share, ~cleared & ~referred, p),
    }
    clusters = joined["split_group"].to_numpy()
    for name, (metric, a, b) in estimates.items():
        result[name] = metric(a, b)
        result[f"{name}_low"], result[f"{name}_high"] = bootstrap_ci(
            metric, a, b, clusters=clusters, n_resamples=n_resamples, seed=seed
        )
    result["refer_precision"] = float(referable[referred].mean()) if referred.any() else np.nan
    decided = cleared | referred
    result["decided_accuracy"] = (
        float(np.mean(referred[decided] == referable[decided])) if decided.any() else np.nan
    )
    return result


def accuracy_coverage(
    labels: npt.ArrayLike, p: npt.ArrayLike, coverages: npt.ArrayLike = COVERAGES
) -> pd.DataFrame:
    """Accuracy of ``p >= 0.5`` on the most confident share of images, for each coverage."""
    y = np.asarray(labels, dtype=bool)
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(-np.abs(p - 0.5), kind="stable")
    correct = ((p >= 0.5) == y)[order]
    rows = []
    for coverage in np.asarray(coverages, dtype=np.float64):
        kept = max(1, round(coverage * len(y)))
        rows.append({"coverage": coverage, "accuracy": float(correct[:kept].mean())})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args(argv)

    targets = OmegaConf.load(args.policy)
    manifest = pd.read_parquet(args.manifest)
    raw = pd.read_parquet(args.predictions)

    val = _join(fitting_rows(manifest, "val"), raw)
    t_referable = fit_binary_temperature(val["p_referable"], val["dr_grade"].astype(int) >= 2)
    t_grade = fit_class_temperature(val[PROBABILITY_COLUMNS], val["dr_grade"].astype(int))
    calibrated = calibrate_predictions(raw, t_referable, t_grade)

    calib = _join(fitting_rows(manifest, "calib"), calibrated)
    calib_referable = (calib["dr_grade"].astype(int) >= 2).to_numpy()
    low = lambda_low(calib.loc[calib_referable, "p_referable"], float(targets.alpha))
    high = max(
        low, lambda_high(calib_referable, calib["p_referable"], float(targets.min_precision))
    )
    thresholds = Thresholds(lambda_low=low, lambda_high=high, temperature_referable=t_referable)

    sets = {"eyepacs_calib": fitting_rows(manifest, "calib")} | {
        name: rows for name, rows in evaluation_sets(manifest).items() if not rows.empty
    }
    calibration, deferral, reliability, coverage = {}, {}, [], []
    for name, rows in sets.items():
        before, after = _join(rows, raw), _join(rows, calibrated)
        calibration[name] = {
            f"{k}_{stage}": v
            for stage, joined in (("before", before), ("after", after))
            for k, v in calibration_metrics(joined).items()
        }
        deferral[name] = deferral_metrics(after, thresholds, args.resamples)
        referable = after["dr_grade"].astype(int) >= 2
        for stage, joined in (("before", before), ("after", after)):
            table = reliability_table(referable, joined["p_referable"])
            reliability.append(table.assign(set=name, head="referable", stage=stage))
            correct, confidence = _top_label(joined)
            table = reliability_table(correct, confidence)
            reliability.append(table.assign(set=name, head="grade", stage=stage))
        coverage.append(accuracy_coverage(referable, after["p_referable"]).assign(set=name))

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    policy = thresholds.to_dict() | {
        "temperature_grade": t_grade,
        "alpha": float(targets.alpha),
        "min_precision": float(targets.min_precision),
        "calib_images": len(calib),
        "calib_referable": int(calib_referable.sum()),
        "predictions": str(args.predictions),
    }
    (out / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    calibrated.to_parquet(out / "predictions.parquet", index=False)
    calibration_table = pd.DataFrame.from_dict(calibration, orient="index")
    deferral_table = pd.DataFrame.from_dict(deferral, orient="index")
    calibration_table.to_csv(out / "calibration.csv")
    deferral_table.to_csv(out / "deferral.csv")
    pd.concat(reliability, ignore_index=True).to_csv(out / "reliability.csv", index=False)
    pd.concat(coverage, ignore_index=True).to_csv(out / "coverage.csv", index=False)

    pd.set_option("display.width", 200)
    print(
        f"T referable {t_referable:.3f}, T grade {t_grade:.3f}; "
        f"lambda_low {low:.4f}, lambda_high {high:.4f} "
        f"({int(calib_referable.sum())} referable of {len(calib)} calib images)"
    )
    print(calibration_table.round(4).to_string())
    print(deferral_table.round(3).to_string())


if __name__ == "__main__":
    main()
