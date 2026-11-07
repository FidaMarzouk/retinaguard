"""Grader metrics on plain arrays, and bootstrap confidence intervals."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score, roc_curve

# Rate comparisons allow for float rounding (1 - 0.9 is 0.09999999999999998).
TOLERANCE = 1e-9


def _two_classes(labels: npt.ArrayLike) -> bool:
    return bool(np.unique(np.asarray(labels)).size == 2)


def auroc(labels: npt.ArrayLike, scores: npt.ArrayLike) -> float:
    """Area under the ROC curve; NaN when only one class is present."""
    if not _two_classes(labels):
        return float("nan")
    return float(roc_auc_score(np.asarray(labels), np.asarray(scores)))


def auprc(labels: npt.ArrayLike, scores: npt.ArrayLike) -> float:
    """Area under the precision-recall curve (average precision); NaN with one class."""
    if not _two_classes(labels):
        return float("nan")
    return float(average_precision_score(np.asarray(labels), np.asarray(scores)))


def sensitivity_at_specificity(
    labels: npt.ArrayLike, scores: npt.ArrayLike, specificity: float = 0.9
) -> float:
    """Highest sensitivity of any threshold whose specificity is at least ``specificity``."""
    if not _two_classes(labels):
        return float("nan")
    fpr, tpr, _ = roc_curve(np.asarray(labels), np.asarray(scores))
    return float(tpr[fpr <= 1 - specificity + TOLERANCE].max())


def specificity_at_sensitivity(
    labels: npt.ArrayLike, scores: npt.ArrayLike, sensitivity: float = 0.95
) -> float:
    """Highest specificity of any threshold whose sensitivity is at least ``sensitivity``."""
    if not _two_classes(labels):
        return float("nan")
    fpr, tpr, _ = roc_curve(np.asarray(labels), np.asarray(scores))
    return float(1 - fpr[tpr >= sensitivity - TOLERANCE].min())


def quadratic_kappa(grades: npt.ArrayLike, predicted: npt.ArrayLike) -> float:
    """Quadratic weighted Cohen's kappa between true and predicted ICDR grades; NaN when only
    one grade occurs in both (kappa is undefined)."""
    if np.unique(np.r_[np.asarray(grades), np.asarray(predicted)]).size < 2:
        return float("nan")
    return float(cohen_kappa_score(np.asarray(grades), np.asarray(predicted), weights="quadratic"))


def accuracy(grades: npt.ArrayLike, predicted: npt.ArrayLike) -> float:
    return float(np.mean(np.asarray(grades) == np.asarray(predicted)))


def bootstrap_ci(
    metric: Callable[[npt.NDArray[np.generic], npt.NDArray[np.generic]], float],
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    clusters: npt.ArrayLike | None = None,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval of ``metric(a, b)``.

    With ``clusters`` (e.g. a patient or duplicate group per image), whole clusters are
    resampled, so correlated images (two eyes of one patient) do not narrow the interval.
    Resamples where the metric is undefined (one class only) are skipped.
    """
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    if clusters is None:
        members = None
        n_units = len(a)
    else:
        codes = pd.factorize(np.asarray(clusters))[0]
        order = np.argsort(codes, kind="stable")
        members = np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)
        n_units = len(members)
    values = np.empty(n_resamples)
    for i in range(n_resamples):
        units = rng.integers(0, n_units, n_units)
        index = units if members is None else np.concatenate([members[u] for u in units])
        values[i] = metric(a[index], b[index])
    alpha = (1 - level) / 2
    low, high = np.nanquantile(values, [alpha, 1 - alpha])
    return float(low), float(high)
