"""Exact (sha256) and perceptual (DCT phash) image hashes for duplicate detection."""

import hashlib

import cv2
import numpy as np

from retinaguard.data.preprocess import Image

PHASH_SIZE = 8  # 8x8 low-frequency DCT block -> 64-bit hash
PHASH_HIGHFREQ_FACTOR = 4  # DCT computed on a 32x32 downscale
# Hamming distance at or below which two images count as near-duplicates (PLAN.md §7.2, inv. 3).
NEAR_DUPLICATE_MAX_DISTANCE = 4


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def phash(img: Image) -> str:
    """64-bit perceptual hash as 16 hex characters.

    Grey 32x32 downscale, 2-D DCT, keep the top-left 8x8 low frequencies, threshold at their
    median. Robust to resizing, JPEG re-encoding and mild brightness changes.
    """
    side = PHASH_SIZE * PHASH_HIGHFREQ_FACTOR
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(grey, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:PHASH_SIZE, :PHASH_SIZE]
    bits = (low > np.median(low)).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()
