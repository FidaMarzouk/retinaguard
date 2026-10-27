"""Synthetic fundus-like images shared by the unit tests."""

import cv2
import numpy as np

from retinaguard.data.preprocess import Image

FUNDUS_RGB = (180, 90, 40)  # orange-red, like a real fundus


def synthetic_fundus(
    width: int = 800, height: int = 600, cx: int = 400, cy: int = 300, radius: int = 250
) -> Image:
    """Orange disc on black with a darker 'optic disc' spot and mild JPEG-like noise."""
    rng = np.random.default_rng(0)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(img, (cx, cy), radius, FUNDUS_RGB, thickness=-1)
    cv2.circle(img, (cx - radius // 2, cy), radius // 8, (240, 200, 150), thickness=-1)
    noise = rng.integers(0, 4, img.shape, dtype=np.uint8)
    return np.asarray(cv2.add(img, noise), dtype=np.uint8)


def textured_fundus(seed: int = 0) -> Image:
    """Synthetic fundus with vessel-like lines and lesion-like blobs, for perceptual hashing."""
    rng = np.random.default_rng(seed)
    img = synthetic_fundus()
    for _ in range(12):
        start = tuple(int(v) for v in rng.integers([200, 80], [600, 520]))
        end = tuple(int(v) for v in rng.integers([200, 80], [600, 520]))
        cv2.line(img, start, end, (110, 30, 20), thickness=int(rng.integers(2, 7)))
    for _ in range(40):
        centre = tuple(int(v) for v in rng.integers([220, 100], [580, 500]))
        cv2.circle(img, centre, int(rng.integers(3, 14)), (220, 180, 90), thickness=-1)
    return img


def lookalike_fundus(seed: int = 0) -> Image:
    """Same disc and macula layout for every seed; only the faint vessel pattern differs."""
    rng = np.random.default_rng(seed)
    img = synthetic_fundus()
    cv2.circle(img, (430, 300), 40, (120, 50, 20), thickness=-1)
    for _ in range(40):
        start = tuple(int(v) for v in rng.integers([200, 80], [600, 520]))
        end = tuple(int(v) for v in rng.integers([200, 80], [600, 520]))
        cv2.line(img, start, end, (165, 78, 35), thickness=2)
    return img
