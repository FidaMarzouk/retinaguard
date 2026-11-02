"""Photometric statistics of a processed fundus image, to describe sites and monitor drift.

Measured on ``preprocess`` outputs (fundus crop, before local mean normalisation), inside the
fundus only, so the black frame and how much of it shows do not count. Luminance is Rec. 601
(0.299 R + 0.587 G + 0.114 B) on the 0-255 scale.
"""

import cv2
import numpy as np

from retinaguard.data.preprocess import Image, circular_mask, fundus_mask

STATS_VERSION = "v1"
DARK_LUMINANCE = 20
BRIGHT_LUMINANCE = 235
# Sharpness ignores this fraction of the radius along the fundus edge, where the step from fundus
# to black frame would dominate the Laplacian.
EDGE_MARGIN_FRACTION = 0.05
STAT_NAMES = (
    "brightness",
    "contrast",
    "red",
    "green",
    "blue",
    "dark_fraction",
    "bright_fraction",
    "sharpness",
    "visible_disc",
)


def image_stats(img: Image) -> dict[str, float]:
    """Brightness (mean luminance), contrast (its standard deviation), mean of each channel,
    share of under- and over-exposed fundus pixels, sharpness (variance of the luminance
    Laplacian) and the share of the fundus disc inside the frame."""
    mask = fundus_mask(img)
    pixels = img[mask].astype(np.float32)
    luminance = pixels @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    size = img.shape[0]
    margin = max(1, round(EDGE_MARGIN_FRACTION * size / 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    inner = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)

    return {
        "brightness": float(luminance.mean()),
        "contrast": float(luminance.std()),
        "red": float(pixels[:, 0].mean()),
        "green": float(pixels[:, 1].mean()),
        "blue": float(pixels[:, 2].mean()),
        "dark_fraction": float((luminance < DARK_LUMINANCE).mean()),
        "bright_fraction": float((luminance > BRIGHT_LUMINANCE).mean()),
        "sharpness": float(laplacian[inner].var()) if inner.any() else 0.0,
        "visible_disc": float(mask.sum() / circular_mask(size, size / 2).sum()),
    }
