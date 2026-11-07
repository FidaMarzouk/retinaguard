"""Grader metrics on plain arrays."""

import numpy as np
import numpy.typing as npt
from sklearn.metrics import cohen_kappa_score, roc_auc_score


def auroc(labels: npt.ArrayLike, scores: npt.ArrayLike) -> float:
    """Area under the ROC curve; NaN when only one class is present."""
    y = np.asarray(labels)
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores)))


def quadratic_kappa(grades: npt.ArrayLike, predicted: npt.ArrayLike) -> float:
    """Quadratic weighted Cohen's kappa between true and predicted ICDR grades."""
    return float(cohen_kappa_score(np.asarray(grades), np.asarray(predicted), weights="quadratic"))
