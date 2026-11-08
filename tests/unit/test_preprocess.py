import cv2
import numpy as np
import pytest
from fundus_fixtures import FUNDUS_RGB, synthetic_fundus

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
    surround_colour,
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


def _on_canvas(colour: tuple[int, int, int], width: int = 1200, height: int = 800) -> Image:
    """A 250 px disc in the middle of a uniform canvas, as some cameras save it."""
    img = np.full((height, width, 3), colour, np.uint8)
    disc = synthetic_fundus(width=500, height=500, cx=250, cy=250, radius=250)
    inside = circular_mask(500, 249)
    top, left = (height - 500) // 2, (width - 500) // 2
    img[top : top + 500, left : left + 500][inside] = disc[inside]
    return img


@pytest.mark.parametrize("colour", [(30, 30, 30), (255, 255, 255)], ids=["grey", "white"])
def test_find_fundus_on_canvas(colour: tuple[int, int, int]) -> None:
    img = _on_canvas(colour)
    assert surround_colour(img) is not None
    circle = find_fundus(img)
    assert circle.cx == pytest.approx(600, abs=3)
    assert circle.cy == pytest.approx(400, abs=3)
    assert circle.radius == pytest.approx(250, abs=3)


def test_find_fundus_white_canvas_with_black_ring() -> None:
    # white page, black camera mask around the disc
    img = np.full((800, 1200, 3), 255, np.uint8)
    cv2.circle(img, (600, 400), 300, (0, 0, 0), thickness=-1)
    cv2.circle(img, (600, 400), 250, FUNDUS_RGB, thickness=-1)
    assert find_fundus(img).radius == pytest.approx(250, abs=3)


def test_find_fundus_two_discs_keeps_larger() -> None:
    img = np.full((600, 1400, 3), 25, np.uint8)
    cv2.circle(img, (350, 300), 250, FUNDUS_RGB, thickness=-1)
    cv2.circle(img, (1050, 300), 200, FUNDUS_RGB, thickness=-1)
    circle = find_fundus(img)
    assert circle.cx == pytest.approx(350, abs=3)
    assert circle.radius == pytest.approx(250, abs=3)


def test_black_surround_is_not_a_canvas() -> None:
    assert surround_colour(synthetic_fundus()) is None
    # clipped disc: fundus reaches the left and right edges, corners still black
    assert surround_colour(synthetic_fundus(width=900, height=400, cx=450, cy=200)) is None


def test_preprocess_on_canvas_fills_square() -> None:
    out, _ = preprocess(_on_canvas((40, 40, 40)))
    assert out[256, 300, 0] > 150  # fundus colour in the middle
    assert out[256, 20, 0] > 150  # and reaching the side, not a small disc in a grey frame
    assert (out[:10, :10] < 10).all()  # canvas painted black like a normal surround


def test_preprocess_two_discs_drops_the_second() -> None:
    img = np.full((600, 1400, 3), 25, np.uint8)
    cv2.circle(img, (350, 300), 250, FUNDUS_RGB, thickness=-1)
    # a smaller disc reaching into the bottom-right corner of the first disc's crop square
    cv2.circle(img, (700, 560), 150, FUNDUS_RGB, thickness=-1)
    out, _ = preprocess(img)
    assert (out[-4:, -4:] < 10).all()


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


def test_local_mean_normalise_near_black_image() -> None:
    # e.g. a dark processed image further darkened by augmentation: no fundus edge to find
    img = np.full((512, 512, 3), 3, dtype=np.uint8)
    with pytest.raises(FundusNotFoundError):
        fundus_mask(img)
    out = local_mean_normalise(img)
    assert (out[~circular_mask(512, 0.9 * 256)] == 128).all()


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
