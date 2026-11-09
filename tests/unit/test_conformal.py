import numpy as np
import pytest

from retinaguard.models.conformal import lambda_high, lambda_low


def test_lambda_low_allows_the_bounded_number_of_misses() -> None:
    scores = np.linspace(0.05, 0.95, 19)
    # alpha (n + 1) - 1 = 1 case may fall below: the threshold is the second smallest score
    assert lambda_low(scores, alpha=0.1) == pytest.approx(scores[1])
    assert lambda_low(scores[::-1], alpha=0.1) == pytest.approx(scores[1])


def test_lambda_low_clears_nothing_with_too_few_cases() -> None:
    assert lambda_low(np.linspace(0.1, 0.9, 10), alpha=0.05) == 0.0


def test_lambda_low_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="no referable"):
        lambda_low([], alpha=0.05)
    with pytest.raises(ValueError, match="alpha"):
        lambda_low([0.5], alpha=0)


@pytest.mark.parametrize(("alpha", "n_calib"), [(0.05, 200), (0.05, 1000), (0.1, 60)])
def test_false_negative_rate_bound_holds_over_random_draws(alpha: float, n_calib: int) -> None:
    """Averaged over calibration draws, the share of new referable cases cleared is at most
    alpha, and not far below it (the bound is tight for continuous scores)."""
    rng = np.random.default_rng(0)
    realised = []
    for _ in range(2000):
        calib = rng.beta(4, 2, n_calib)
        new = rng.beta(4, 2, 2000)
        realised.append(np.mean(new < lambda_low(calib, alpha)))
    mean = float(np.mean(realised))
    assert mean <= alpha + 0.002
    assert mean >= alpha - 1 / (n_calib + 1) - 0.002


def test_lambda_high_reaches_target_precision() -> None:
    labels = np.array([0, 0, 1, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    # precision of score >= 0.4 is 4/5; of score >= 0.3 is 5/6, the smallest reaching 0.8
    assert lambda_high(labels, scores, 0.8) == pytest.approx(0.3)
    assert lambda_high(labels, scores, 1.0) == pytest.approx(0.5)
    assert lambda_high(np.zeros(4), np.linspace(0, 1, 4), 0.8) == np.inf


def test_lambda_high_treats_tied_scores_together() -> None:
    labels = np.array([1, 0, 1])
    scores = np.array([0.9, 0.9, 0.1])
    # 0.9 gives precision 1/2 for both tied images (never 1 for the first one), 0.1 gives 2/3
    assert lambda_high(labels, scores, 0.7) == np.inf
