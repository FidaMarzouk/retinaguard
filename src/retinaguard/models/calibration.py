"""Temperature scaling and calibration metrics, on stored probabilities.

A single temperature ``T`` divides a head's logits: ``T > 1`` softens over-confident
probabilities, ``T < 1`` sharpens them. It is fitted on held-out predictions by minimising the
negative log-likelihood, and it never changes the order of images, so AUROC is unaffected.
Logits are recovered from probabilities: exactly for the referable head's sigmoid
(``log(p / (1 - p))``), and up to a per-image constant, which softmax ignores, for the grade
head (``log p``).
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

EPS = 1e-7
N_BINS = 15
# Search range of the temperature; a fitted value at either end means the inputs are odd.
T_MIN, T_MAX = 0.05, 20.0

FloatArray = npt.NDArray[np.float64]


def binary_logits(p: npt.ArrayLike) -> FloatArray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.asarray(np.log(p) - np.log1p(-p), dtype=np.float64)


def class_logits(probs: npt.ArrayLike) -> FloatArray:
    return np.log(np.clip(np.asarray(probs, dtype=np.float64), EPS, 1.0))


def sigmoid(z: FloatArray) -> FloatArray:
    return np.asarray(0.5 * (1 + np.tanh(0.5 * z)), dtype=np.float64)


def softmax(z: FloatArray) -> FloatArray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return np.asarray(e / e.sum(axis=1, keepdims=True), dtype=np.float64)


def scale_binary(p: npt.ArrayLike, temperature: float) -> FloatArray:
    """Referable probabilities after temperature scaling."""
    return sigmoid(binary_logits(p) / temperature)


def scale_classes(probs: npt.ArrayLike, temperature: float) -> FloatArray:
    """Grade probabilities (one row per image) after temperature scaling."""
    return softmax(class_logits(probs) / temperature)


def _minimise_log_scale(loss: Callable[[float], float], iterations: int = 80) -> float:
    """Golden-section search for the temperature in [T_MIN, T_MAX], on a log scale.

    The negative log-likelihood is convex in ``1 / T``, so it has a single minimum here.
    """
    ratio = (np.sqrt(5) - 1) / 2
    low, high = np.log(T_MIN), np.log(T_MAX)
    a, b = high - ratio * (high - low), low + ratio * (high - low)
    loss_a, loss_b = loss(float(np.exp(a))), loss(float(np.exp(b)))
    for _ in range(iterations):
        if loss_a < loss_b:
            high, b, loss_b = b, a, loss_a
            a = high - ratio * (high - low)
            loss_a = loss(float(np.exp(a)))
        else:
            low, a, loss_a = a, b, loss_b
            b = low + ratio * (high - low)
            loss_b = loss(float(np.exp(b)))
    return float(np.exp((low + high) / 2))


def fit_binary_temperature(p: npt.ArrayLike, labels: npt.ArrayLike) -> float:
    """Temperature minimising the binary negative log-likelihood of ``labels`` (0/1)."""
    z = binary_logits(p)
    y = np.asarray(labels, dtype=np.float64)

    def nll(t: float) -> float:
        # log(1 + exp(z/t)) - y * z/t, written to stay finite for large |z|
        s = z / t
        return float(np.mean(np.logaddexp(0, s) - y * s))

    return _minimise_log_scale(nll)


def fit_class_temperature(probs: npt.ArrayLike, grades: npt.ArrayLike) -> float:
    """Temperature minimising the multiclass negative log-likelihood of ``grades``."""
    z = class_logits(probs)
    rows = np.arange(len(z))
    y = np.asarray(grades, dtype=np.int64)

    def nll(t: float) -> float:
        s = z / t
        log_norm = np.logaddexp.reduce(s, axis=1)
        return float(np.mean(log_norm - s[rows, y]))

    return _minimise_log_scale(nll)


def reliability_table(
    labels: npt.ArrayLike, confidence: npt.ArrayLike, n_bins: int = N_BINS
) -> pd.DataFrame:
    """Equal-width bins of ``confidence`` in [0, 1] with their observed rate of ``labels``.

    For the referable head, ``confidence`` is ``p_referable`` and ``labels`` the referable
    truth. For the grade head (top-label calibration), ``confidence`` is the largest grade
    probability and ``labels`` whether that grade is correct. Empty bins are left out.
    """
    y = np.asarray(labels, dtype=np.float64)
    c = np.asarray(confidence, dtype=np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(c, edges[1:-1]), 0, n_bins - 1)
    table = (
        pd.DataFrame({"bin": bins, "confidence": c, "label": y})
        .groupby("bin")
        .agg(
            images=("label", "size"),
            mean_confidence=("confidence", "mean"),
            observed=("label", "mean"),
        )
    )
    table.insert(0, "bin_high", edges[table.index + 1])
    table.insert(0, "bin_low", edges[table.index])
    return table.reset_index(drop=True)


def expected_calibration_error(
    labels: npt.ArrayLike, confidence: npt.ArrayLike, n_bins: int = N_BINS
) -> float:
    """Image-weighted mean gap between confidence and observed rate over the bins."""
    table = reliability_table(labels, confidence, n_bins)
    if table.empty:
        return float("nan")
    gaps = (table["observed"] - table["mean_confidence"]).abs()
    return float((gaps * table["images"]).sum() / table["images"].sum())


def brier_score(labels: npt.ArrayLike, p: npt.ArrayLike) -> float:
    """Mean squared error of binary probabilities."""
    return float(np.mean((np.asarray(p, dtype=np.float64) - np.asarray(labels)) ** 2))


def multiclass_brier_score(grades: npt.ArrayLike, probs: npt.ArrayLike) -> float:
    """Mean over images of the squared error summed over classes."""
    probs = np.asarray(probs, dtype=np.float64)
    one_hot = np.eye(probs.shape[1])[np.asarray(grades, dtype=np.int64)]
    return float(np.mean(((probs - one_hot) ** 2).sum(axis=1)))
