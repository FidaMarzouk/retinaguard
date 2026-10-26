import cv2
import numpy as np
from fundus_fixtures import synthetic_fundus, textured_fundus

from retinaguard.data.hashing import NEAR_DUPLICATE_MAX_DISTANCE, hamming, phash, sha256_hex
from retinaguard.data.preprocess import decode_rgb, encode_jpeg, preprocess


def test_sha256_hex() -> None:
    assert sha256_hex(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_phash_format() -> None:
    h = phash(synthetic_fundus())
    assert len(h) == 16
    int(h, 16)


def test_phash_stable_under_resize_and_jpeg() -> None:
    original = preprocess(textured_fundus())[0]
    resized = cv2.resize(cv2.resize(original, (300, 300)), (512, 512))
    recompressed = decode_rgb(encode_jpeg(original, quality=70))
    assert hamming(phash(original), phash(resized)) <= NEAR_DUPLICATE_MAX_DISTANCE
    assert hamming(phash(original), phash(recompressed)) <= NEAR_DUPLICATE_MAX_DISTANCE


def test_phash_separates_different_images() -> None:
    a = preprocess(textured_fundus(seed=0))[0]
    b = preprocess(textured_fundus(seed=1))[0]
    assert hamming(phash(a), phash(b)) > NEAR_DUPLICATE_MAX_DISTANCE


def test_hamming() -> None:
    assert hamming("0000000000000000", "0000000000000000") == 0
    assert hamming("0000000000000000", "ffffffffffffffff") == 64
    assert hamming("0f00000000000000", "0000000000000001") == 5


def test_phash_ignores_uniform_brightness_shift() -> None:
    img = preprocess(textured_fundus())[0]
    brighter = np.clip(img.astype(int) + 15, 0, 255).astype(np.uint8)
    assert hamming(phash(img), phash(brighter)) <= NEAR_DUPLICATE_MAX_DISTANCE
