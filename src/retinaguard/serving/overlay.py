"""Lesion overlays: predicted lesion regions drawn on the fundus image, one colour per type.

Regions above the threshold are tinted and outlined; the image stays visible underneath. The
colours (Okabe-Ito, distinguishable with colour-vision deficiencies, and chosen to stand out on
an orange-red retina) are fixed per lesion type, and ``LEGEND`` names them for the app.
"""

import cv2
import numpy as np
import numpy.typing as npt

from retinaguard.data.segmentation import LESION_NAMES, LESIONS

COLOURS: dict[str, tuple[int, int, int]] = {  # RGB
    "MA": (86, 180, 233),
    "HE": (0, 158, 115),
    "EX": (240, 228, 66),
    "SE": (204, 121, 167),
}
LEGEND = {LESION_NAMES[lesion]: COLOURS[lesion] for lesion in LESIONS}
THRESHOLD = 0.5
ALPHA = 0.45


def overlay(
    image: npt.NDArray[np.uint8],
    probs: npt.NDArray[np.floating],
    threshold: float = THRESHOLD,
    alpha: float = ALPHA,
) -> npt.NDArray[np.uint8]:
    """RGB image with lesion regions of ``probs`` (lesions, H, W) tinted and outlined.

    ``probs`` may be at a lower resolution than ``image``; it is resized to match.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an RGB image, got shape {image.shape}")
    if probs.shape[0] != len(LESIONS):
        raise ValueError(f"expected {len(LESIONS)} lesion channels, got {probs.shape[0]}")
    height, width = image.shape[:2]
    out = image.astype(np.float32)
    outlines = []
    for lesion, channel in zip(LESIONS, probs, strict=True):
        if channel.shape != (height, width):
            channel = cv2.resize(
                channel.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR
            )
        region = channel >= threshold
        if not region.any():
            continue
        tint = np.array(COLOURS[lesion], dtype=np.float32)
        out[region] = (1 - alpha) * out[region] + alpha * tint
        contours, _ = cv2.findContours(
            region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        outlines.append((contours, COLOURS[lesion]))
    result = np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8))
    for contours, colour in outlines:
        cv2.drawContours(result, contours, -1, colour, thickness=1)
    return result


def lesion_summary(
    probs: npt.NDArray[np.floating], threshold: float = THRESHOLD
) -> dict[str, float]:
    """Share of image pixels predicted as each lesion type, keyed by lesion name."""
    return {
        LESION_NAMES[lesion]: float((p >= threshold).mean())
        for lesion, p in zip(LESIONS, probs, strict=True)
    }
