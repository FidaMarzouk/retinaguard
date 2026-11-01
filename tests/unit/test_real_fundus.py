"""Preprocessing and duplicate detection on real fundus photographs.

The fixtures are four IDRiD disease grading images (Porwal et al., IEEE DataPort,
https://doi.org/10.21227/H25W98, CC BY 4.0), downscaled from 4288x2848 to 1072x712.
"""

from itertools import combinations
from pathlib import Path

import cv2
import pytest

from retinaguard.data.duplicates import (
    CANDIDATE_MAX_DISTANCE,
    DUPLICATE_MIN_SIMILARITY,
    structure_signature,
    structure_similarity,
)
from retinaguard.data.hashing import hamming, phash
from retinaguard.data.preprocess import (
    DEFAULT_SIZE,
    Image,
    decode_rgb,
    encode_jpeg,
    find_fundus,
    preprocess,
)
from retinaguard.data.processing import process_bytes

FIXTURES = Path(__file__).parents[1] / "fixtures" / "idrid"
# training-set grades: two healthy eyes, exudates (2), large haemorrhage (4)
GRADES = {"IDRiD_005": 4, "IDRiD_016": 2, "IDRiD_138": 0, "IDRiD_139": 0}


def _load(name: str) -> Image:
    return decode_rgb((FIXTURES / f"{name}.jpg").read_bytes())


@pytest.fixture(scope="module")
def processed() -> dict[str, Image]:
    return {name: preprocess(_load(name))[0] for name in GRADES}


@pytest.mark.parametrize("name", GRADES)
def test_fundus_found_like_full_dataset(name: str) -> None:
    img = _load(name)
    height, width = img.shape[:2]
    circle = find_fundus(img)
    # All IDRiD images: fundus diameter ~0.80 of the width, clipped top and bottom.
    assert 2 * circle.radius / width == pytest.approx(0.81, abs=0.03)
    assert circle.cy / height == pytest.approx(0.5, abs=0.02)
    assert 0.4 < circle.cx / width < 0.55


@pytest.mark.parametrize("name", GRADES)
def test_preprocess_output(name: str, processed: dict[str, Image]) -> None:
    out = processed[name]
    assert out.shape == (DEFAULT_SIZE, DEFAULT_SIZE, 3)
    assert out[:8, :8].max() < 20  # corner outside the fundus stays black
    centre = out[DEFAULT_SIZE // 2 - 16 : DEFAULT_SIZE // 2 + 16, DEFAULT_SIZE // 2]
    assert centre[:, 0].mean() > 60  # red channel of the retina


def test_process_bytes_on_real_file() -> None:
    raw = (FIXTURES / "IDRiD_016.jpg").read_bytes()
    record, jpeg = process_bytes("idrid:training_IDRiD_016", raw)
    assert record.error is None
    assert (record.width, record.height) == (1072, 712)
    assert jpeg is not None
    assert record.phash is not None


@pytest.mark.parametrize("name", GRADES)
def test_degraded_copy_is_duplicate(name: str, processed: dict[str, Image]) -> None:
    brighter = cv2.convertScaleAbs(_load(name), alpha=1.1)
    copy = preprocess(decode_rgb(encode_jpeg(brighter, quality=75)))[0]
    original = processed[name]
    assert hamming(phash(original), phash(copy)) <= CANDIDATE_MAX_DISTANCE
    similarity = structure_similarity(structure_signature(original), structure_signature(copy))
    assert similarity >= DUPLICATE_MIN_SIMILARITY + 0.2


@pytest.mark.parametrize(("a", "b"), list(combinations(GRADES, 2)))
def test_different_eyes_are_not_duplicates(a: str, b: str, processed: dict[str, Image]) -> None:
    sig_a, sig_b = (structure_signature(processed[n]) for n in (a, b))
    assert structure_similarity(sig_a, sig_b) < DUPLICATE_MIN_SIMILARITY - 0.3
