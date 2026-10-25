"""Fundus image preprocessing (PLAN.md §8.1).

Stored images are only cropped to the fundus and resized (``PREPROCESS_VERSION``). Ben Graham
normalisation is applied on the fly by ``ben_graham`` so one processed dataset serves both arms
of the with/without ablation. Images are RGB ``uint8`` arrays of shape (H, W, 3).

Blur width and border mask follow Graham's competition report (see
``docs/references/graham-2015-kaggle-dr.md``): sigma = fundus radius / 30, and the outer 10 % of
the fundus is masked to grey to hide the edge ring the blur creates.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

Image = npt.NDArray[np.uint8]

DEFAULT_SIZE = 512
PREPROCESS_VERSION = f"crop{DEFAULT_SIZE}_v1"

# A pixel belongs to the fundus if its red channel exceeds this fraction of the image's mean red
# (Graham uses mean / 10 on the summed channels). Relative, so dark images are still found; the
# absolute floor keeps JPEG noise in a near-black background from counting as fundus.
RED_THRESHOLD_FRACTION = 0.1
RED_THRESHOLD_FLOOR = 10
MIN_RADIUS_PX = 32
BEN_GRAHAM_BLUR_DIVISOR = 30
BEN_GRAHAM_MASK_FRACTION = 0.9
GREY = 128


class FundusNotFoundError(ValueError):
    """No plausible fundus disc in the image (blank, over-exposed or not a fundus photo)."""


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    radius: float


def find_fundus(img: Image) -> Circle:
    """Locate the fundus disc as the largest bright region of the red channel.

    The radius comes from the region's widest extent, so discs clipped at the top and bottom of
    the frame (common in EyePACS) still get their true radius.
    """
    _check_rgb(img)
    red = img[:, :, 0].astype(np.float32)
    threshold = max(red.mean() * RED_THRESHOLD_FRACTION, RED_THRESHOLD_FLOOR)
    mask = (red > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise FundusNotFoundError("no bright region in the red channel")
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    radius = max(w, h) / 2
    if radius < MIN_RADIUS_PX:
        raise FundusNotFoundError(f"fundus radius {radius:.0f}px below {MIN_RADIUS_PX}px")
    return Circle(cx=x + w / 2, cy=y + h / 2, radius=radius)


def crop_to_fundus(img: Image, circle: Circle) -> Image:
    """Square crop of side 2·radius centred on the disc, black-padded where it leaves the frame."""
    side = round(2 * circle.radius)
    left = round(circle.cx - circle.radius)
    top = round(circle.cy - circle.radius)
    out = np.zeros((side, side, 3), dtype=np.uint8)
    src_x0, src_y0 = max(left, 0), max(top, 0)
    src_x1, src_y1 = min(left + side, img.shape[1]), min(top + side, img.shape[0])
    out[src_y0 - top : src_y1 - top, src_x0 - left : src_x1 - left] = img[
        src_y0:src_y1, src_x0:src_x1
    ]
    return out


def resize_square(img: Image, size: int = DEFAULT_SIZE) -> Image:
    interpolation = cv2.INTER_AREA if img.shape[0] >= size else cv2.INTER_CUBIC
    return np.asarray(cv2.resize(img, (size, size), interpolation=interpolation), dtype=np.uint8)


def preprocess(img: Image, size: int = DEFAULT_SIZE) -> tuple[Image, Circle]:
    """Crop to the fundus and resize; returns the processed image and the disc in raw pixels."""
    circle = find_fundus(img)
    return resize_square(crop_to_fundus(img, circle), size), circle


def circular_mask(size: int, radius: float) -> npt.NDArray[np.bool_]:
    yy, xx = np.ogrid[:size, :size]
    centre = (size - 1) / 2
    return np.asarray((xx - centre) ** 2 + (yy - centre) ** 2 <= radius**2)


def ben_graham(img: Image) -> Image:
    """Subtract the local mean colour (mapped to grey) and mask the fundus border.

    Expects a ``preprocess`` output, where the fundus fills the square (radius = size / 2).
    """
    _check_rgb(img)
    size = img.shape[0]
    radius = size / 2
    blurred = cv2.GaussianBlur(img, (0, 0), radius / BEN_GRAHAM_BLUR_DIVISOR)
    out = np.asarray(cv2.addWeighted(img, 4, blurred, -4, GREY), dtype=np.uint8)
    out[~circular_mask(size, BEN_GRAHAM_MASK_FRACTION * radius)] = GREY
    return out


def _check_rgb(img: Image) -> None:
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError(f"expected an RGB uint8 image, got shape {img.shape} {img.dtype}")
