from pathlib import Path

import numpy as np
import pytest

from retinaguard.models.ood import Mahalanobis, energy, ood_threshold


def _clusters(rng: np.random.Generator, n: int, dim: int = 8) -> tuple[np.ndarray, np.ndarray]:
    labels = rng.integers(0, 3, n)
    centres = np.eye(3, dim) * 5
    return centres[labels] + rng.normal(size=(n, dim)), labels


def test_mahalanobis_scores_unusual_points_higher(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x, y = _clusters(rng, 3000)
    detector = Mahalanobis.fit(x, y)
    inside, _ = _clusters(rng, 2000)
    outside = rng.normal(size=(500, 8)) + 15  # far from every class mean
    threshold = ood_threshold(detector.score(inside), 99)
    assert np.mean(detector.score(inside) > threshold) == pytest.approx(0.01, abs=0.002)
    assert np.mean(detector.score(outside) > threshold) == 1.0
    # a point at a class mean scores ~0, whichever class
    assert detector.score(detector.means).max() < 1e-6

    detector.save(tmp_path / "ood.npz")
    loaded = Mahalanobis.load(tmp_path / "ood.npz")
    assert np.allclose(loaded.score(outside[:5]), detector.score(outside[:5]))


def test_energy_is_lower_for_confident_logits() -> None:
    confident, unsure = [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    scores = energy([confident, unsure])
    assert scores[0] < scores[1]
    assert scores[1] == pytest.approx(-np.log(3))
