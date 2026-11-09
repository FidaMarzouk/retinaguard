import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retinaguard.models.calibration import sigmoid
from retinaguard.serving.policy import Thresholds
from retinaguard.train.calibrate import accuracy_coverage, deferral_metrics, main
from retinaguard.train.fit_ood import main as fit_ood_main
from retinaguard.train.predict import PROBABILITY_COLUMNS


def _synthetic(n_per_split: int = 1500) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Manifest and over-confident predictions for EyePACS val/calib/test and one site."""
    rng = np.random.default_rng(0)
    splits = [("eyepacs", "val"), ("eyepacs", "calib"), ("eyepacs", "test_internal")]
    splits.append(("idrid", "test_external"))
    frames = []
    for source, split in splits:
        n = n_per_split
        z = rng.normal(0, 2.5, n)
        grade = np.where(rng.random(n) < sigmoid(z), rng.integers(2, 5, n), rng.integers(0, 2, n))
        ids = [f"{source}:{split}_{i}" for i in range(n)]
        frames.append(
            pd.DataFrame(
                {
                    "image_id": ids,
                    "source": source,
                    "split": split,
                    "split_group": ids,
                    "dr_grade": pd.array(grade, dtype="Int64"),
                    "referable": pd.array(grade >= 2, dtype="boolean"),
                    "quality_label": None,
                    "processed_path": "p",
                    "_z": z,
                }
            )
        )
    manifest = pd.concat(frames, ignore_index=True)
    grade_probs = np.full((len(manifest), 5), 0.05)
    grade_probs[np.arange(len(manifest)), manifest["dr_grade"].astype(int)] = 0.8
    predictions = pd.DataFrame(grade_probs, columns=PROBABILITY_COLUMNS).assign(
        image_id=manifest["image_id"], p_referable=sigmoid(3 * manifest["_z"].to_numpy())
    )
    return manifest.drop(columns="_z"), predictions


def test_calibrate_main(tmp_path: Path) -> None:
    manifest, predictions = _synthetic()
    manifest.to_parquet(tmp_path / "manifest.parquet")
    predictions.to_parquet(tmp_path / "predictions.parquet")
    out = tmp_path / "policy"
    main(
        [
            f"--manifest={tmp_path / 'manifest.parquet'}",
            f"--predictions={tmp_path / 'predictions.parquet'}",
            f"--out-dir={out}",
            "--resamples=50",
        ]
    )
    policy = json.loads((out / "policy.json").read_text())
    assert policy["temperature_referable"] == pytest.approx(3, rel=0.2)
    thresholds = Thresholds.load(out / "policy.json")
    assert 0 < thresholds.lambda_low < thresholds.lambda_high < 1

    calibration = pd.read_csv(out / "calibration.csv", index_col=0)
    assert set(calibration.index) == {
        "eyepacs_calib",
        "eyepacs_test",
        "eyepacs_test_all",
        "idrid",
        "idrid_all",
    }
    assert (calibration["referable_ece_after"] < calibration["referable_ece_before"]).all()

    deferral = pd.read_csv(out / "deferral.csv", index_col=0)
    assert deferral.loc["eyepacs_test", "fnr"] < 0.05 + 0.03
    shares = deferral[["no_refer_rate", "refer_rate", "defer_rate"]].sum(axis=1)
    assert np.allclose(shares, 1)
    assert len(pd.read_parquet(out / "predictions.parquet")) == len(predictions)
    assert set(pd.read_csv(out / "reliability.csv")["head"]) == {"referable", "grade"}
    assert pd.read_csv(out / "coverage.csv")["coverage"].max() == 1.0


def test_deferral_metrics_counts_decisions() -> None:
    joined = pd.DataFrame(
        {
            "dr_grade": pd.array([0, 0, 2, 3, 4, 1], dtype="Int64"),
            "p_referable": [0.01, 0.5, 0.02, 0.5, 0.95, 0.9],
            "split_group": list("abcdef"),
        }
    )
    result = deferral_metrics(joined, Thresholds(lambda_low=0.1, lambda_high=0.8), n_resamples=20)
    assert result["fnr"] == pytest.approx(1 / 3)
    assert result["no_refer_rate"] == pytest.approx(2 / 6)
    assert result["refer_rate"] == pytest.approx(2 / 6)
    assert result["defer_rate"] == pytest.approx(2 / 6)
    assert result["refer_precision"] == pytest.approx(0.5)
    assert result["decided_accuracy"] == pytest.approx(2 / 4)


def test_accuracy_coverage_drops_least_confident_first() -> None:
    labels = np.array([1, 0, 1, 0])
    p = np.array([0.99, 0.01, 0.45, 0.55])  # the two least confident are wrong
    curve = accuracy_coverage(labels, p, [0.5, 1.0])
    assert curve["accuracy"].tolist() == [1.0, 0.5]


def test_fit_ood_main(tmp_path: Path) -> None:
    manifest, predictions = _synthetic(400)
    manifest.to_parquet(tmp_path / "manifest.parquet")
    run = tmp_path / "run"
    run.mkdir()
    predictions.to_parquet(run / "predictions.parquet")
    rng = np.random.default_rng(1)
    embeddings = rng.normal(size=(len(predictions), 6))
    embeddings[manifest["source"].to_numpy() == "idrid"] += 3  # the external site looks different
    np.save(run / "embeddings.npy", embeddings.astype(np.float16))
    policy_dir = run / "policy"
    main(
        [
            f"--manifest={tmp_path / 'manifest.parquet'}",
            f"--predictions={run / 'predictions.parquet'}",
            f"--out-dir={policy_dir}",
            "--resamples=20",
        ]
    )
    fit_ood_main(
        [
            f"--manifest={tmp_path / 'manifest.parquet'}",
            f"--predictions-dir={run}",
            f"--policy-dir={policy_dir}",
            "--resamples=20",
        ]
    )
    thresholds = Thresholds.load(policy_dir / "policy.json")
    assert thresholds.ood_threshold is not None
    assert json.loads((policy_dir / "policy.json").read_text())["ood_percentile"] == 99
    table = pd.read_csv(policy_dir / "ood.csv", index_col=0)
    assert table.loc["eyepacs_calib", "flagged"] == pytest.approx(0.01, abs=0.01)
    assert table.loc["idrid", "flagged"] > 0.5
    assert table.loc["idrid", "site_auroc"] > 0.9
