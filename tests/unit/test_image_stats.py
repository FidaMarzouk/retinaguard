from pathlib import Path

import cv2
import numpy as np
import pytest
from fundus_fixtures import FUNDUS_RGB, synthetic_fundus, textured_fundus

from retinaguard.data.image_stats import STAT_NAMES, image_stats
from retinaguard.data.preprocess import decode_rgb, preprocess

FIXTURE = Path(__file__).parents[1] / "fixtures" / "idrid" / "IDRiD_005.jpg"


def _processed(img: np.ndarray) -> np.ndarray:
    return preprocess(img)[0]


def test_stats_measure_the_fundus_only() -> None:
    stats = image_stats(_processed(synthetic_fundus()))
    assert tuple(stats) == STAT_NAMES
    expected = 0.299 * FUNDUS_RGB[0] + 0.587 * FUNDUS_RGB[1] + 0.114 * FUNDUS_RGB[2]
    # the black frame is excluded; the bright optic-disc spot lifts the mean a little
    assert stats["brightness"] == pytest.approx(expected, abs=8)
    assert stats["red"] > stats["green"] > stats["blue"]
    assert stats["visible_disc"] == pytest.approx(1.0, abs=0.03)
    assert stats["dark_fraction"] < 0.001  # only anti-aliased edge pixels


def test_darker_blurrier_and_clipped_images_score_lower() -> None:
    img = _processed(textured_fundus())
    base = image_stats(img)
    darker = image_stats(np.asarray(cv2.convertScaleAbs(img, alpha=0.5), dtype=np.uint8))
    blurred = image_stats(np.asarray(cv2.GaussianBlur(img, (0, 0), 3), dtype=np.uint8))
    clipped = synthetic_fundus(height=420)  # disc cut at top and bottom by the frame
    assert darker["brightness"] == pytest.approx(base["brightness"] / 2, rel=0.05)
    assert darker["contrast"] < base["contrast"]
    assert blurred["sharpness"] < base["sharpness"] / 4
    assert image_stats(_processed(clipped))["visible_disc"] < 0.85


def test_real_fundus() -> None:
    stats = image_stats(_processed(decode_rgb(FIXTURE.read_bytes())))
    assert 40 < stats["brightness"] < 200
    assert stats["red"] > stats["blue"]
    assert 0.7 < stats["visible_disc"] <= 1.0
    assert stats["sharpness"] > 0
