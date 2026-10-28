"""Near-duplicate detection for processed fundus images.

A 64-bit phash only sees the coarse layout (disc position, dark macula, black frame), which
most fundus photos share, so different eyes often get the same hash. It is therefore used only
to propose candidate pairs; a pair counts as a duplicate when the fine structure (vessels,
lesions) also matches, measured by ``structure_similarity``.
"""

from collections.abc import Sequence

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd

from retinaguard.data.preprocess import Image, circular_mask, local_mean_normalise

# Hamming distance at or below which two phashes make a candidate pair.
CANDIDATE_MAX_DISTANCE = 4
SIGNATURE_SIZE = 64
SIGNATURE_MASK_FRACTION = 0.85
# Calibrated on 1,000 EyePACS images: re-encoded copies score >= 0.53 (p01 0.80), random pairs
# of different patients <= 0.30. Set low because wrongly grouping two patients into one split
# costs almost nothing, while a missed duplicate leaks an eye across splits.
DUPLICATE_MIN_SIMILARITY = 0.5

Signature = npt.NDArray[np.float32]

_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hash_values(hashes: Sequence[str]) -> npt.NDArray[np.uint64]:
    return np.array([int(h, 16) for h in hashes], dtype=np.uint64)


def candidate_pairs(
    hashes: Sequence[str], max_distance: int = CANDIDATE_MAX_DISTANCE, block: int = 256
) -> npt.NDArray[np.int64]:
    """Index pairs (i < j) whose phashes differ by at most ``max_distance`` bits, shape (n, 2)."""
    values = hash_values(hashes)
    found = [np.empty((0, 2), dtype=np.int64)]
    for start in range(0, len(values), block):
        rows = values[start : start + block]
        rest = values[start:]
        xor = rows[:, None] ^ rest[None, :]
        distance = _POPCOUNT8[xor.view(np.uint8)].reshape(*xor.shape, 8).sum(axis=-1)
        i, j = np.nonzero(distance <= max_distance)
        keep = i < j
        found.append(np.stack([i[keep] + start, j[keep] + start], axis=1).astype(np.int64))
    return np.concatenate(found)


def structure_signature(img: Image) -> Signature:
    """Zero-mean, unit-norm vector of the normalised green channel at 64 px inside the fundus.

    Expects a ``preprocess`` output. Local mean normalisation removes illumination and colour
    cast, so what remains is vessels and lesions; the outer fundus border is left out.
    """
    green = local_mean_normalise(img)[:, :, 1]
    small = cv2.resize(green, (SIGNATURE_SIZE, SIGNATURE_SIZE), interpolation=cv2.INTER_AREA)
    inside = circular_mask(SIGNATURE_SIZE, SIGNATURE_MASK_FRACTION * SIGNATURE_SIZE / 2)
    values = small[inside].astype(np.float32)
    values -= values.mean()
    norm = float(np.linalg.norm(values))
    return values / norm if norm > 0 else values


def structure_similarity(a: Signature, b: Signature) -> float:
    """Pearson correlation of two signatures, in [-1, 1]; 0 if either image is featureless."""
    return float(np.dot(a, b))


def is_duplicate(a: Signature, b: Signature) -> bool:
    return structure_similarity(a, b) >= DUPLICATE_MIN_SIMILARITY


def confirmed_pairs(
    candidates: pd.DataFrame, min_similarity: float = DUPLICATE_MIN_SIMILARITY
) -> list[tuple[str, str]]:
    """``(image_id_a, image_id_b)`` of candidate pairs whose structure similarity passes."""
    kept = candidates[candidates["similarity"] >= min_similarity]
    return list(zip(kept["image_id_a"], kept["image_id_b"], strict=True))
