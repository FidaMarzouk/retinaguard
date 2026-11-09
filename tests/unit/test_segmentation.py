from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from retinaguard.data.segmentation import (
    LESIONS,
    SegShards,
    ddr_pairs,
    pack,
    prepare_pair,
    read_index,
    unpack,
)
from retinaguard.models.segmentation import (
    PixelStats,
    SegLoss,
    aupr,
    concat_stats,
    dice,
    image_stats,
    seg_model,
)
from retinaguard.serving.overlay import COLOURS, lesion_summary, overlay
from retinaguard.train.evaluate_seg import main as evaluate_main
from retinaguard.train.train_seg import SegDataset, lesion_crop, site_scores


def test_pack_round_trip() -> None:
    masks = np.random.default_rng(0).random((4, 16, 16)) > 0.7
    assert np.array_equal(unpack(pack(masks)), masks)


def test_prepare_pair_moves_masks_with_the_crop() -> None:
    img = np.zeros((600, 800, 3), np.uint8)
    cv2.circle(img, (500, 300), 250, (180, 90, 40), -1)
    masks = np.zeros((4, 600, 800), bool)
    masks[0, 299:301, 549:551] = True  # a 2x2 microaneurysm 50 px right of the centre
    masks[2, 200:260, 400:460] = True
    image, resized = prepare_pair(img, masks, size=256)
    assert image.shape == (256, 256, 3)
    ys, xs = np.nonzero(resized[0])
    assert len(ys) > 0  # survives a ~2x downscale
    # the disc spans 500 px -> 256 px; the dot is ~50 px right of centre -> ~26 px
    assert abs(xs.mean() - (128 + 50 * 256 / 500)) < 3
    assert abs(ys.mean() - 128) < 3
    assert not resized[1].any()
    assert not resized[3].any()
    with pytest.raises(ValueError, match="mask shape"):
        prepare_pair(img, masks[:, :10], size=256)


def test_seg_shard_index_and_reader(seg_shard: Path) -> None:
    index = read_index([seg_shard])
    assert index.groupby("split").size().to_dict() == {"test": 4, "train": 8, "val": 2}
    assert (index["pixels_MA"] > 0).all()
    assert (index["pixels_SE"] > 0).sum() == 7  # odd images only
    image, masks = SegShards(index).read(index["image_id"].iloc[0])
    assert image.shape == (128, 128, 3)
    assert masks.shape == (4, 128, 128)


def test_ddr_pairs_reads_release_layout(tmp_path: Path) -> None:
    for split, annotation in (("train", "train"), ("val", "val"), ("test", "tet")):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "images" / split / "007-1-100.jpg").write_bytes(b"")
        for lesion in LESIONS:
            folder = tmp_path / "annotations" / annotation / lesion
            folder.mkdir(parents=True)
            (folder / "007-1-100.tif").write_bytes(b"")
    pairs = ddr_pairs(tmp_path)
    assert [p.split for p in pairs] == ["train", "val", "test"]
    assert pairs[2].image_id == "ddr_seg:007-1-100"
    assert pairs[2].mask_paths["SE"] == tmp_path / "annotations" / "tet" / "SE" / "007-1-100.tif"
    (tmp_path / "annotations" / "tet" / "MA" / "007-1-100.tif").unlink()
    with pytest.raises(FileNotFoundError, match="missing DDR mask"):
        ddr_pairs(tmp_path)


def _stats(probs: np.ndarray, masks: np.ndarray) -> PixelStats:
    return image_stats(probs, masks)


def test_dice_and_aupr() -> None:
    rng = np.random.default_rng(0)
    masks = rng.random((4, 64, 64)) > 0.9
    masks[3] = False  # one lesion type absent
    perfect = _stats(masks.astype(np.float32), masks)
    assert np.allclose(dice(perfect)[:3], 1)
    assert np.allclose(aupr(perfect)[:3], 1)
    assert np.isnan(aupr(perfect)[3])
    assert np.isnan(dice(perfect)[3])
    random = _stats(rng.random((4, 64, 64)), masks)
    assert np.allclose(aupr(random)[:3], masks[:3].mean(axis=(1, 2)), atol=0.03)
    both = concat_stats([perfect, random])
    assert both.tp.shape == (2, 4)
    assert np.array_equal(both.take([0]).tp, perfect.tp)


def test_seg_model_and_loss() -> None:
    model = seg_model("resnet18", pretrained=False)
    logits = model(torch.zeros(2, 3, 64, 64))
    assert logits.shape == (2, 4, 64, 64)
    loss = SegLoss()(logits, torch.zeros(2, 4, 64, 64))
    assert torch.isfinite(loss)


def test_lesion_crop_centres_on_a_lesion() -> None:
    masks = np.zeros((4, 200, 200), bool)
    masks[1, 190, 10] = True
    top, left = lesion_crop(masks, 64, np.random.default_rng(0), on_lesion=True)
    assert (top, left) == (136, 0)  # clipped to the frame, lesion inside
    top, left = lesion_crop(np.zeros_like(masks), 64, np.random.default_rng(0), on_lesion=True)
    assert 0 <= top <= 136
    assert 0 <= left <= 136


def test_seg_dataset_crops_and_full_images(seg_shard: Path) -> None:
    index = read_index([seg_shard])
    image, masks = SegDataset(index, crop_size=64, lesion_crop_fraction=1.0)[0]
    assert image.shape == (3, 64, 64)
    assert masks.shape == (4, 64, 64)
    assert masks.sum() > 0
    image, masks = SegDataset(index)[0]
    assert image.shape == (3, 128, 128)


def test_site_scores_average_sites() -> None:
    rows = pd.DataFrame({"site": ["a", "b"]})
    masks = np.zeros((4, 32, 32), bool)
    masks[:, :4, :4] = True
    good = _stats(masks.astype(np.float32), masks)
    flat = _stats(np.full((4, 32, 32), 0.4), masks)
    scores = site_scores(rows, concat_stats([good, flat]))
    assert scores["val_aupr_a_MA"] == pytest.approx(1)
    assert scores["val_score"] == pytest.approx((1 + masks[0].mean()) / 2)


def test_overlay_tints_only_lesions() -> None:
    image = np.full((40, 40, 3), 100, np.uint8)
    probs = np.zeros((4, 20, 20))
    probs[2, 5:10, 5:10] = 0.9  # hard exudates, at half resolution
    out = overlay(image, probs)
    assert out.shape == image.shape
    assert (out[20:, 20:] == 100).all()
    blended = (1 - 0.45) * 100 + 0.45 * np.array(COLOURS["EX"])
    assert np.allclose(out[15, 15], blended, atol=2)
    assert lesion_summary(probs)["hard exudates"] == pytest.approx(25 / 400)
    with pytest.raises(ValueError, match="lesion channels"):
        overlay(image, probs[:2])


def _write_run(folder: Path, sites: list[str], probs_noise: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    parts, rows = [], []
    for site in ("idrid", "ddr"):
        for i in range(12):
            masks = rng.random((4, 24, 24)) > 0.9
            probs = np.clip(masks + rng.normal(0, probs_noise, masks.shape), 0, 1)
            parts.append(image_stats(probs, masks))
            rows.append((f"{site}:{i}", site, "test"))
    stats = concat_stats(parts)
    folder.mkdir(parents=True)
    ids, sites_col, splits = map(np.asarray, zip(*rows, strict=True))
    np.savez_compressed(
        folder / "stats.npz",
        image_id=ids,
        site=sites_col,
        split=splits,
        tp=stats.tp,
        fp=stats.fp,
        fn=stats.fn,
        positive_hist=stats.positive_hist,
        negative_hist=stats.negative_hist,
    )
    (folder / "config.yaml").write_text(f"data:\n  train_sites: {sites}\n")


def test_evaluate_seg_main(tmp_path: Path) -> None:
    _write_run(tmp_path / "seg-pooled", ["idrid", "ddr"], 0.2, 0)
    _write_run(tmp_path / "seg-ddr_only", ["ddr"], 0.6, 0)
    out = tmp_path / "eval"
    evaluate_main(
        [
            str(tmp_path / "seg-pooled"),
            str(tmp_path / "seg-ddr_only"),
            "--reference=seg-pooled",
            f"--out-dir={out}",
            "--resamples=50",
        ]
    )
    metrics = pd.read_csv(out / "metrics.csv")
    assert set(metrics["set"]) == {"idrid", "ddr", "idrid_all"}
    assert ((metrics["low"] <= metrics["value"]) & (metrics["value"] <= metrics["high"])).all()
    comparison = pd.read_csv(out / "comparison.csv")
    aupr_diff = comparison[comparison["metric"] == "aupr"]
    assert (aupr_diff["difference"] < 0).all()  # noisier predictions are worse
    assert (aupr_diff["high"] < 0).all()
    sites = pd.read_csv(out / "sites.csv")
    assert (sites["gap"] >= 0).all()
