"""Thresholds of the deferral policy, chosen on the ``calib`` split.

An image is cleared (``NO_REFER``) when its referable score is below ``lambda_low`` and
referred when it is at least ``lambda_high``; anything between goes to a human.

``lambda_low`` uses conformal risk control on the referable ``calib`` cases, with the loss
``1[p < lambda]`` (a referable eye wrongly cleared). It is the largest threshold with

    n / (n + 1) * share_below(lambda) + 1 / (n + 1) <= alpha,

which bounds the expected false-negative rate (the share of referable eyes cleared) on new
images from the same distribution at ``alpha``. The guarantee is on average over calibration
draws, and only holds while new images resemble the calibration ones, so it says nothing
about a different site; the realised rate there must be measured.
"""

import numpy as np
import numpy.typing as npt

from retinaguard.models.metrics import TOLERANCE


def lambda_low(referable_scores: npt.ArrayLike, alpha: float = 0.05) -> float:
    """Clearing threshold from the scores of the referable calibration cases.

    Returns 0.0 (clear nothing) when there are too few cases for the bound, that is when
    ``n + 1 < 1 / alpha``.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    scores = np.sort(np.asarray(referable_scores, dtype=np.float64))
    n = len(scores)
    if n == 0:
        raise ValueError("no referable calibration cases")
    # The condition allows at most k referable cases strictly below the threshold, and
    # scores[k] is the largest threshold with that property.
    k = int(np.floor(alpha * (n + 1) - 1 + TOLERANCE))
    if k < 0:
        return 0.0
    return float(scores[k]) if k < n else 1.0


def lambda_high(labels: npt.ArrayLike, scores: npt.ArrayLike, min_precision: float = 0.80) -> float:
    """Smallest threshold whose referrals (``score >= threshold``) reach ``min_precision``.

    Returns ``inf`` (refer nothing automatically) when no threshold reaches it.
    """
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    # precision of "score >= s[i]", evaluated at the last position of each distinct score
    last = np.r_[s[1:] != s[:-1], True]
    precision = np.cumsum(y)[last] / (np.flatnonzero(last) + 1)
    reached = s[last][precision >= min_precision - TOLERANCE]
    return float(reached.min()) if reached.size else float("inf")
