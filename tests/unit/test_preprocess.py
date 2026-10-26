import cv2
import numpy as np
import pytest
from fundus_fixtures import synthetic_fundus

from retinaguard.data.preprocess import (
    Circle,
    FundusNotFoundError,
    Image,
    circular_mask,
    crop_to_fundus,
    decode_rgb,
    encode_jpeg,
    find_fundus,
    fundus_mask,
    local_mean_normalise,
    preprocess,
)


def test_find_fundus_centred() -> None:
    circle = find_fundus(synthetic_fundus())
    assert circle.cx == pytest.approx(400, abs=2)
    assert circle.cy == pytest.approx(300, abs=2)
    assert circle.radius == pytest.approx(250, abs=2)


def test_find_fundus_clipped_top_and_bottom() -> None:
    # Disc taller than the frame, as in many EyePACS images: radius from the full width.
    circle = find_fundus(synthetic_fundus(width=900, height=400, cx=450, cy=200, radius=300))
    assert circle.radius == pytest.approx(300, abs=2)
    assert circle.cx == pytest.approx(450, abs=2)


def test_find_fundus_dark_image() -> None:
    dark = (synthetic_fundus().astype(np.float32) * 0.15).astype(np.uint8)
    assert find_fundus(dark).radius == pytest.approx(250, abs=3)


@pytest.mark.parametrize(
    "img",
    [np.zeros((300, 300, 3), np.uint8), synthetic_fundus(radius=10)],
    ids=["blank", "tiny"],
)
def test_find_fundus_rejects(img: Image) -> None:
    with pytest.raises(FundusNotFoundError):
        find_fundus(img)


def test_rejects_non_rgb() -> None:
    with pytest.raises(ValueError, match="RGB uint8"):
        find_fundus(np.zeros((100, 100), np.uint8))


def test_crop_pads_outside_frame() -> None:
    img = np.full((100, 100, 3), 200, np.uint8)
    out = crop_to_fundus(img, Circle(cx=50, cy=20, radius=50))
    assert out.shape == (100, 100, 3)
    assert (out[:30] == 0).all()  # rows above the frame are padding
    assert (out[30:] == 200).all()


def test_preprocess_output() -> None:
    out, circle = preprocess(synthetic_fundus(width=1200, height=900, cx=650, cy=450, radius=420))
    assert out.shape == (512, 512, 3)
    assert out.dtype == np.uint8
    assert circle.radius == pytest.approx(420, abs=2)
    # Fundus fills the square: centre is fundus-coloured, corners are background.
    assert out[300, 300, 0] > 150
    assert (out[:10, :10] < 10).all()


def test_preprocess_upscales_small_images() -> None:
    out, _ = preprocess(synthetic_fundus(width=300, height=300, cx=150, cy=150, radius=120))
    assert out.shape == (512, 512, 3)


def test_local_mean_normalise_flat_region_maps_to_grey() -> None:
    out = local_mean_normalise(preprocess(synthetic_fundus())[0])
    # Uniform fundus interior (away from the optic disc) becomes ~128 after local-mean removal;
    # averaged over a patch because the x4 gain amplifies per-pixel noise.
    patch = out[236:276, 310:350].reshape(-1, 3).mean(axis=0)
    assert np.abs(patch - 128).max() <= 3


def test_local_mean_normalise_masks_border() -> None:
    out = local_mean_normalise(preprocess(synthetic_fundus())[0])
    outside = ~circular_mask(512, 0.9 * 256)
    assert (out[outside] == 128).all()
    assert out.shape == (512, 512, 3)


def test_jpeg_round_trip_keeps_rgb_order() -> None:
    img = np.zeros((64, 64, 3), np.uint8)
    img[..., 0] = 200  # pure red must stay in channel 0
    decoded = decode_rgb(encode_jpeg(img))
    assert decoded.shape == img.shape
    assert np.abs(decoded.astype(int) - img).max() <= 3


def test_decode_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="decode"):
        decode_rgb(b"not an image")


def test_local_mean_normalise_masks_clipped_edges() -> None:
    # Disc cut off at top and bottom: the crop pads with black, which must not leave stripes.
    clipped = synthetic_fundus(width=900, height=400, cx=450, cy=200, radius=300)
    out = local_mean_normalise(preprocess(clipped)[0])
    # Fundus spans rows ~85..426 of the 512 crop; 10 % margin = ~26 px inside each cut.
    assert (out[:105] == 128).all()
    assert (out[-105:] == 128).all()
    assert np.abs(out[256, 330].astype(int) - 128).max() < 40  # interior kept, not masked


def test_fundus_mask_has_no_holes() -> None:
    img = synthetic_fundus()
    cv2.circle(img, (400, 300), 30, (0, 0, 0), thickness=-1)  # very dark macula
    mask = fundus_mask(img)
    assert mask[300, 400]
    assert mask.sum() == pytest.approx(np.pi * 250**2, rel=0.02)


def test_circular_mask_area() -> None:
    mask = circular_mask(512, 256)
    assert mask.sum() == pytest.approx(np.pi * 256**2, rel=0.01)
