import cv2
import numpy as np
import pandas as pd
import pytest
from fundus_fixtures import lookalike_fundus, textured_fundus

from retinaguard.data.duplicates import (
    DUPLICATE_MIN_SIMILARITY,
    candidate_frame,
    candidate_pairs,
    confirmed_pairs,
    is_duplicate,
    score_candidates,
    structure_signature,
    structure_similarity,
)
from retinaguard.data.preprocess import Image, decode_rgb, encode_jpeg, preprocess


def _signature(raw: Image) -> np.ndarray:
    return structure_signature(preprocess(raw)[0])


def test_candidate_pairs_by_hamming_distance() -> None:
    hashes = ["0000000000000000", "000000000000000f", "00000000000000ff", "ffffffffffffffff"]
    pairs = candidate_pairs(hashes, max_distance=4, block=2)
    assert pairs.tolist() == [[0, 1], [1, 2]]


def test_candidate_pairs_handles_top_bit_and_empty_input() -> None:
    assert candidate_pairs(["8000000000000000", "8000000000000000"]).tolist() == [[0, 1]]
    assert candidate_pairs([]).shape == (0, 2)


def test_candidate_pairs_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    base = int(rng.integers(0, 2**63))
    hashes = [
        f"{base ^ (1 << int(b)) ^ (1 << int(c)):016x}" for b, c in rng.integers(0, 64, (50, 2))
    ]
    expected = [
        [i, j]
        for i in range(len(hashes))
        for j in range(i + 1, len(hashes))
        if (int(hashes[i], 16) ^ int(hashes[j], 16)).bit_count() <= 2
    ]
    assert candidate_pairs(hashes, max_distance=2, block=7).tolist() == expected


def test_signature_is_unit_norm_and_zero_mean() -> None:
    sig = _signature(textured_fundus())
    assert sig.dtype == np.float32
    assert np.linalg.norm(sig) == pytest.approx(1.0, abs=1e-5)
    assert float(sig.mean()) == pytest.approx(0.0, abs=1e-5)


def test_re_encoded_copies_are_duplicates() -> None:
    raw = textured_fundus()
    original = _signature(raw)
    fundus = raw[:, :, :1] > 30  # shift the fundus only; the black frame stays black
    copies = {
        "downscaled raw": cv2.resize(raw, (400, 300), interpolation=cv2.INTER_AREA),
        "jpeg 70": decode_rgb(encode_jpeg(raw, quality=70)),
        "brighter": np.clip(raw.astype(int) + 15 * fundus, 0, 255).astype(np.uint8),
        "more contrast": np.clip(raw.astype(float) * 1.2, 0, 255).astype(np.uint8),
        "padded frame": np.pad(raw, ((0, 0), (6, 0), (0, 0))),
    }
    for name, copy in copies.items():
        assert is_duplicate(original, _signature(copy)), name


def test_different_images_are_not_duplicates() -> None:
    a, b = _signature(textured_fundus(seed=0)), _signature(textured_fundus(seed=1))
    assert structure_similarity(a, b) < DUPLICATE_MIN_SIMILARITY - 0.2


def test_shared_layout_alone_scores_below_copies() -> None:
    a, b = _signature(lookalike_fundus(seed=0)), _signature(lookalike_fundus(seed=1))
    assert structure_similarity(a, b) < 0.9


def test_candidate_frame_with_distances_and_source_filter() -> None:
    ids = ["eyepacs:1_left", "eyepacs:2_left", "aptos:a", "aptos:b"]
    hashes = ["0000000000000000", "0000000000000001", "0000000000000003", "ffffffffffffffff"]

    frame = candidate_frame(ids, hashes, max_distance=2)
    assert frame.to_dict("list") == {
        "image_id_a": ["eyepacs:1_left", "eyepacs:1_left", "eyepacs:2_left"],
        "image_id_b": ["eyepacs:2_left", "aptos:a", "aptos:a"],
        "phash_distance": [1, 2, 1],
    }
    only_aptos = candidate_frame(ids, hashes, max_distance=2, involving="aptos")
    assert only_aptos["image_id_b"].tolist() == ["aptos:a", "aptos:a"]


def test_score_candidates() -> None:
    signatures = {
        "x": _signature(textured_fundus(seed=0)),
        "y": _signature(textured_fundus(seed=1)),
    }
    pairs = pd.DataFrame({"image_id_a": ["x", "x"], "image_id_b": ["x", "y"]})
    scored = score_candidates(pairs, signatures.__getitem__)
    assert scored["similarity"].iloc[0] == pytest.approx(1.0, abs=1e-5)
    assert scored["similarity"].iloc[1] < DUPLICATE_MIN_SIMILARITY
    assert confirmed_pairs(scored) == [("x", "x")]
