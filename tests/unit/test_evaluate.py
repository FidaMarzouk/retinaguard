import numpy as np
import pandas as pd
import pytest

from retinaguard.models.metrics import (
    auprc,
    auroc,
    bootstrap_ci,
    sensitivity_at_specificity,
    specificity_at_sensitivity,
)
from retinaguard.train.evaluate import confusion_matrix, evaluate, evaluation_sets
from retinaguard.train.predict import PROBABILITY_COLUMNS


def test_operating_point_metrics() -> None:
    labels = np.array([0] * 10 + [1] * 10)
    scores = np.r_[np.linspace(0, 0.5, 10), np.linspace(0.45, 1, 10)]
    # one negative (0.5) scores above the lowest positive (0.45)
    assert sensitivity_at_specificity(labels, scores, 0.9) == 1.0
    assert specificity_at_sensitivity(labels, scores, 1.0) == pytest.approx(0.9)
    assert auprc(labels, labels) == 1.0
    assert np.isnan(specificity_at_sensitivity([0, 0], [0.1, 0.2]))


def test_bootstrap_contains_estimate_and_respects_clusters() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 200)
    scores = labels + rng.normal(0, 1, 200)
    low, high = bootstrap_ci(auroc, labels, scores, n_resamples=300)
    assert low < auroc(labels, scores) < high
    # every image duplicated: as clusters, the interval matches the original data's width
    doubled = (np.repeat(labels, 2), np.repeat(scores, 2))
    naive = bootstrap_ci(auroc, *doubled, n_resamples=300)
    clustered = bootstrap_ci(
        auroc, *doubled, clusters=np.repeat(np.arange(200), 2), n_resamples=300
    )
    assert clustered[1] - clustered[0] > (naive[1] - naive[0]) * 1.2
    assert clustered[1] - clustered[0] == pytest.approx(high - low, rel=0.25)


def _manifest() -> pd.DataFrame:
    rows = [
        # image_id, source, split, split_group, grade, quality
        ("eyepacs:1_left", "eyepacs", "test_internal", "eyepacs:1", 0, None),
        ("eyepacs:1_right", "eyepacs", "test_internal", "eyepacs:1", 2, None),
        ("eyepacs:2_left", "eyepacs", "test_internal", "eyepacs:2", 4, "reject"),
        ("eyepacs:3_left", "eyepacs", "val", "eyepacs:3", 1, None),
        ("ddr:test_a", "ddr", "test_external", "ddr:test_a", 2, None),
        ("ddr:test_b", "ddr", "test_external", "ddr:test_a", 2, None),  # duplicate of a
        ("ddr:test_c", "ddr", "test_external", "ddr:test_c", 0, None),
        ("ddr:test_d", "ddr", "test_external", "ddr:test_c", 3, None),  # referable conflict
        ("ddr:test_e", "ddr", "test_external", "ddr:test_e", None, "reject"),  # ungradable
        ("ddr:test_f", "ddr", "test_external", "ddr:test_f", 1, None),
    ]
    frame = pd.DataFrame(
        rows, columns=["image_id", "source", "split", "split_group", "dr_grade", "quality_label"]
    )
    grade = frame["dr_grade"].astype("Int64")
    return frame.assign(
        dr_grade=grade, referable=(grade >= 2).astype("boolean"), processed_path="p"
    )


def test_evaluation_sets() -> None:
    sets = {name: rows["image_id"].tolist() for name, rows in evaluation_sets(_manifest()).items()}
    assert sets["eyepacs_test"] == ["eyepacs:1_left", "eyepacs:1_right"]
    assert sets["eyepacs_test_all"] == ["eyepacs:1_left", "eyepacs:1_right", "eyepacs:2_left"]
    assert sets["ddr"] == ["ddr:test_a", "ddr:test_f"]
    assert sets["ddr_all"] == ["ddr:test_a", "ddr:test_b", "ddr:test_c", "ddr:test_d", "ddr:test_f"]
    assert sets["aptos"] == []


def test_evaluate_with_perfect_predictions() -> None:
    manifest = _manifest()
    one_hot = np.eye(5)[manifest["dr_grade"].fillna(0).astype(int)]
    predictions = pd.DataFrame(one_hot, columns=PROBABILITY_COLUMNS).assign(
        image_id=manifest["image_id"], p_referable=one_hot[:, 2:].sum(1)
    )
    manifest = manifest[manifest["source"] == "eyepacs"]
    table = evaluate(manifest, predictions, n_resamples=50)
    internal = table.loc["eyepacs_test_all"]
    assert internal["images"] == 3
    assert internal["auroc"] == 1.0
    assert internal["quadratic_kappa"] == 1.0
    rows = evaluation_sets(manifest)["eyepacs_test_all"].merge(predictions, on="image_id")
    assert np.trace(confusion_matrix(rows).to_numpy()) == 3


def test_evaluate_requires_predictions() -> None:
    manifest = _manifest()
    predictions = pd.DataFrame(columns=["image_id", *PROBABILITY_COLUMNS, "p_referable"])
    with pytest.raises(ValueError, match="no prediction"):
        evaluate(manifest, predictions.astype({"image_id": str}), n_resamples=10)
