import numpy as np
import pytest

from retinaguard.models.calibration import (
    T_MAX,
    binary_logits,
    brier_score,
    expected_calibration_error,
    fit_binary_temperature,
    fit_class_temperature,
    multiclass_brier_score,
    reliability_table,
    scale_binary,
    scale_classes,
    sigmoid,
    softmax,
)
from retinaguard.models.metrics import auroc


def test_fit_binary_temperature_recovers_overconfidence() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, 20_000)
    labels = rng.random(z.size) < sigmoid(z)
    overconfident = sigmoid(3 * z)
    t = fit_binary_temperature(overconfident, labels)
    assert t == pytest.approx(3, rel=0.1)
    rescaled = scale_binary(overconfident, t)
    assert expected_calibration_error(labels, rescaled) < 0.02
    assert expected_calibration_error(labels, overconfident) > 0.05
    assert auroc(labels, rescaled) == pytest.approx(auroc(labels, overconfident))


def test_fit_class_temperature_recovers_underconfidence() -> None:
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2, (20_000, 5))
    probs = softmax(z)
    grades = (rng.random(len(z))[:, None] > probs.cumsum(1)).sum(1)
    t = fit_class_temperature(softmax(z / 2), grades)  # softened by T = 2
    assert t == pytest.approx(0.5, rel=0.1)
    assert np.allclose(scale_classes(softmax(z / 2), t).sum(1), 1)


def test_temperature_stays_in_range_for_separable_data() -> None:
    labels = np.r_[np.zeros(50), np.ones(50)]
    assert fit_binary_temperature(np.where(labels == 1, 0.9, 0.1), labels) < 1
    assert fit_binary_temperature(np.full(100, 0.5), labels) <= T_MAX


def test_binary_logits_handle_extremes() -> None:
    assert np.isfinite(binary_logits([0.0, 1.0])).all()
    assert np.allclose(scale_binary([0.2, 0.5, 0.8], 1.0), [0.2, 0.5, 0.8])


def test_reliability_table_and_ece() -> None:
    labels = np.array([0, 0, 0, 1, 1])
    confidence = np.array([0.05, 0.08, 0.95, 0.95, 1.0])
    table = reliability_table(labels, confidence, n_bins=10)
    assert table["images"].tolist() == [2, 3]
    assert table["bin_low"].tolist() == [0.0, 0.9]
    assert table["observed"].tolist() == pytest.approx([0, 2 / 3])
    expected = (2 * 0.065 + 3 * abs(2 / 3 - (0.95 + 0.95 + 1.0) / 3)) / 5
    assert expected_calibration_error(labels, confidence, n_bins=10) == pytest.approx(expected)
    assert expected_calibration_error(np.zeros(4), np.full(4, 0.9)) == pytest.approx(0.9)


def test_brier_scores() -> None:
    assert brier_score([0, 1], [0.0, 1.0]) == 0
    assert brier_score([0, 1], [0.5, 0.5]) == 0.25
    assert multiclass_brier_score([0, 2], np.eye(3)[[0, 2]]) == 0
    assert multiclass_brier_score([0], [[0.5, 0.5, 0.0]]) == pytest.approx(0.5)
