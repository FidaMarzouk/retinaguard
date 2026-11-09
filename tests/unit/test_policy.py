from pathlib import Path

import pytest

from retinaguard.serving.policy import Decision, Outcome, Reason, Thresholds, decide

THRESHOLDS = Thresholds(lambda_low=0.1, lambda_high=0.7)


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        (0.05, Outcome(Decision.NO_REFER)),
        (0.1, Outcome(Decision.DEFER_TO_HUMAN, Reason.UNCERTAIN)),
        (0.5, Outcome(Decision.DEFER_TO_HUMAN, Reason.UNCERTAIN)),
        (0.7, Outcome(Decision.REFER)),
        (1.0, Outcome(Decision.REFER)),
    ],
)
def test_decision_bands(p: float, expected: Outcome) -> None:
    assert decide(p, THRESHOLDS) == expected


def test_quality_reject_comes_first() -> None:
    assert decide(0.99, THRESHOLDS, quality_label="reject") == Outcome(
        Decision.RETAKE, Reason.IMAGE_QUALITY
    )
    assert decide(0.05, THRESHOLDS, quality_label="usable").decision == Decision.NO_REFER


def test_out_of_distribution_defers() -> None:
    thresholds = Thresholds(lambda_low=0.1, lambda_high=0.7, ood_threshold=3.0)
    assert decide(0.05, thresholds, ood_score=5.0) == Outcome(
        Decision.DEFER_TO_HUMAN, Reason.OUT_OF_DISTRIBUTION
    )
    assert decide(0.05, thresholds, ood_score=1.0).decision == Decision.NO_REFER
    # an OOD check that is configured but has no score cannot clear the image
    assert decide(0.05, thresholds).reason == Reason.ERROR


@pytest.mark.parametrize("p", [None, float("nan"), -0.1, 1.5])
def test_invalid_scores_fail_safe(p: float | None) -> None:
    assert decide(p, THRESHOLDS) == Outcome(Decision.DEFER_TO_HUMAN, Reason.ERROR)


def test_errors_while_deciding_fail_safe() -> None:
    assert decide("0.05", THRESHOLDS) == Outcome(  # type: ignore[arg-type]
        Decision.DEFER_TO_HUMAN, Reason.ERROR
    )


def test_thresholds_validate_and_load(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lambda_low"):
        Thresholds(lambda_low=0.8, lambda_high=0.2)
    with pytest.raises(ValueError, match="temperature"):
        Thresholds(lambda_low=0.1, lambda_high=0.2, temperature_referable=0)
    path = tmp_path / "policy.json"
    path.write_text('{"lambda_low": 0.1, "lambda_high": 0.7, "alpha": 0.05}')
    assert Thresholds.load(path) == THRESHOLDS
