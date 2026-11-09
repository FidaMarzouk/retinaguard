"""Out-of-distribution scores: how unlike the reference images an image looks to the grader.

- **Mahalanobis** distance of the pooled backbone embedding to the nearest grade-conditional
  Gaussian (one mean per grade, a shared covariance), so a typical image of any grade scores
  low. The covariance is shrunk towards its diagonal mean for a stable inverse.
- **Energy** of logits, ``-T * logsumexp(z / T)``: higher for inputs the model is less sure
  of. Needs true logits, so it is for serving, not for stored softmax probabilities.

Higher scores mean more unusual; the threshold is a high percentile of in-distribution scores.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

SHRINKAGE = 0.1
PERCENTILE = 99.0


def energy(logits: npt.ArrayLike, temperature: float = 1.0) -> npt.NDArray[np.float64]:
    z = np.asarray(logits, dtype=np.float64) / temperature
    return np.asarray(-temperature * np.logaddexp.reduce(z, axis=1), dtype=np.float64)


@dataclass(frozen=True)
class Mahalanobis:
    means: npt.NDArray[np.float64]  # (classes, D)
    precision: npt.NDArray[np.float64]  # (D, D)

    @classmethod
    def fit(
        cls, embeddings: npt.ArrayLike, labels: npt.ArrayLike, shrinkage: float = SHRINKAGE
    ) -> "Mahalanobis":
        x = np.asarray(embeddings, dtype=np.float64)
        y = np.asarray(labels)
        classes = np.unique(y)
        means = np.stack([x[y == c].mean(axis=0) for c in classes])
        centred = x - means[np.searchsorted(classes, y)]
        covariance = centred.T @ centred / len(x)
        target = np.trace(covariance) / len(covariance) * np.eye(len(covariance))
        shrunk = (1 - shrinkage) * covariance + shrinkage * target
        return cls(means, np.linalg.inv(shrunk))

    def score(self, embeddings: npt.ArrayLike, batch_size: int = 4096) -> npt.NDArray[np.float64]:
        """Distance to the nearest class mean (square root of the quadratic form)."""
        x = np.asarray(embeddings, dtype=np.float64)
        scores = np.empty(len(x))
        for start in range(0, len(x), batch_size):
            diff = x[start : start + batch_size, None, :] - self.means[None]
            quad = ((diff @ self.precision) * diff).sum(axis=2)
            scores[start : start + batch_size] = np.sqrt(np.maximum(quad.min(axis=1), 0))
        return scores

    def save(self, path: Path) -> None:
        np.savez_compressed(path, means=self.means, precision=self.precision)

    @classmethod
    def load(cls, path: Path) -> "Mahalanobis":
        with np.load(path) as data:
            return cls(data["means"], data["precision"])


def ood_threshold(in_distribution_scores: npt.ArrayLike, percentile: float = PERCENTILE) -> float:
    """Score above which an image counts as out of distribution."""
    return float(np.percentile(np.asarray(in_distribution_scores, dtype=np.float64), percentile))
