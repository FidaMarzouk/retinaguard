"""Drift signals between a reference set of predictions and an incoming batch.

- **PSI** (population stability index) of the referable probability, on ten bins at the
  reference's deciles: below 0.1 stable, 0.1-0.25 moderate, above 0.25 a large shift.
- **MMD** (maximum mean discrepancy, RBF kernel, median-distance bandwidth) between grader
  embeddings, with a permutation p-value: how differently the model "sees" the batch.
- **Image statistics:** standardised mean shift of each statistic (in reference SDs).
- **Rates:** retake, deferral, out-of-distribution and referral rates, compared as ratios.

The alert rule (``alerts``) flags a batch whose retake, deferral or out-of-distribution rate
exceeds twice the reference; the other signals are reported to explain an alert, not to raise
one. The retake rate is included because degraded photos (blurred, dark) are mostly sent back
by the quality gate: deferrals then fall, and a rule on deferral alone stays silent.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

PSI_BINS = 10
PSI_EPS = 1e-4
RATE_ALERT_RATIO = 2.0
MMD_MAX_SAMPLES = 500
MMD_PERMUTATIONS = 200
RATES = ("retake_rate", "defer_rate", "ood_rate", "refer_rate")
ALERT_RATES = ("retake_rate", "defer_rate", "ood_rate")


def psi(reference: npt.ArrayLike, current: npt.ArrayLike, bins: int = PSI_BINS) -> float:
    """Population stability index of ``current`` against ``reference``."""
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if not len(ref) or not len(cur):
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    ref_share = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=len(edges) + 1)
    cur_share = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=len(edges) + 1)
    p = np.maximum(ref_share / len(ref), PSI_EPS)
    q = np.maximum(cur_share / len(cur), PSI_EPS)
    return float(np.sum((q - p) * np.log(q / p)))


def _sq_distances(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    out = (a**2).sum(1)[:, None] + (b**2).sum(1)[None] - 2 * a @ b.T
    return np.asarray(np.maximum(out, 0), dtype=np.float64)


def mmd(
    reference: npt.ArrayLike,
    current: npt.ArrayLike,
    *,
    max_samples: int = MMD_MAX_SAMPLES,
    permutations: int = MMD_PERMUTATIONS,
    seed: int = 0,
) -> tuple[float, float]:
    """Squared MMD (biased estimate) and its permutation p-value. Both sets are subsampled to
    ``max_samples`` and standardised with the reference's mean and SD."""
    rng = np.random.default_rng(seed)
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(current, dtype=np.float64)
    x = x[rng.permutation(len(x))[:max_samples]]
    y = y[rng.permutation(len(y))[:max_samples]]
    mean, sd = x.mean(0), x.std(0) + 1e-8
    z = np.concatenate([(x - mean) / sd, (y - mean) / sd])
    d2 = _sq_distances(z, z)
    bandwidth = np.median(d2[np.triu_indices(len(z), 1)])
    kernel = np.exp(-d2 / max(bandwidth, 1e-12))
    n = len(x)

    def statistic(order: npt.NDArray[np.intp]) -> float:
        k = kernel[np.ix_(order, order)]
        return float(k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean())

    observed = statistic(np.arange(len(z)))
    null = [statistic(rng.permutation(len(z))) for _ in range(permutations)]
    p_value = (1 + sum(v >= observed for v in null)) / (1 + permutations)
    return observed, float(p_value)


def mean_shift(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, float]:
    """Per column: difference of means in reference standard deviations."""
    sd = reference.std().replace(0, np.nan)
    shift = (current.mean() - reference.mean()) / sd
    return {str(k): float(v) for k, v in shift.items()}


def decision_rates(decisions: pd.Series, ood_flags: pd.Series | None = None) -> dict[str, float]:
    rates = {
        "retake_rate": float((decisions == "RETAKE").mean()),
        "defer_rate": float((decisions == "DEFER_TO_HUMAN").mean()),
        "refer_rate": float((decisions == "REFER").mean()),
    }
    rates["ood_rate"] = float(ood_flags.mean()) if ood_flags is not None else float("nan")
    return rates


@dataclass(frozen=True)
class Alert:
    signal: str
    value: float
    reference: float

    def __str__(self) -> str:
        limit = f"{RATE_ALERT_RATIO:g} x reference {self.reference:.1%}"
        return f"{self.signal} {self.value:.1%} > {limit}"


def alerts(rates: Mapping[str, float], reference: Mapping[str, float]) -> list[Alert]:
    """Retake, deferral or out-of-distribution rate above ``RATE_ALERT_RATIO`` times the
    reference."""
    raised = []
    for signal in ALERT_RATES:
        value, base = rates.get(signal, np.nan), reference.get(signal, np.nan)
        if np.isfinite(value) and np.isfinite(base) and value > RATE_ALERT_RATIO * base:
            raised.append(Alert(signal, float(value), float(base)))
    return raised
