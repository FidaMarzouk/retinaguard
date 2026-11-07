from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from retinaguard.data.datasets import (
    IMAGENET_MEAN,
    FundusDataset,
    grade_balancing_sampler,
    select_rows,
    to_input,
    train_augment,
)


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": ["e:5", "e:4", "e:3", "e:2", "e:1", "a:1"],
            "split": ["train", "train", "train", "train", "val", "test_external"],
            "processed_path": ["p", "p", None, "p", "p", "p"],
            "dr_grade": pd.array([0, 2, 1, None, 4, 3], dtype="Int64"),
            "quality_label": ["reject", None, "good", None, "usable", None],
        }
    )


def test_select_rows() -> None:
    m = _manifest()
    # e:3 failed preprocessing, e:2 has no grade, e:5 is a quality reject
    assert select_rows(m, ["train"])["image_id"].tolist() == ["e:4"]
    kept = select_rows(m, ["train"], exclude_quality_reject=False)
    assert kept["image_id"].tolist() == ["e:4", "e:5"]
    assert select_rows(m, ["train", "val"], limit_per_split=1)["image_id"].tolist() == [
        "e:1",
        "e:4",
    ]


def test_to_input_resizes_and_scales() -> None:
    img = np.full((512, 512, 3), 255, dtype=np.uint8)
    tensor = to_input(img, size=128, local_mean_norm=False)
    assert tensor.shape == (3, 128, 128)
    assert tensor.dtype == torch.float32
    assert tensor[0, 0, 0].item() == pytest.approx((1 - IMAGENET_MEAN[0]) / 0.229, rel=1e-5)


def test_dataset_reads_shards(fixture_shard: tuple[Path, pd.DataFrame]) -> None:
    shard_dir, manifest = fixture_shard
    rows = select_rows(manifest, ["val"])
    dataset = FundusDataset(rows, [shard_dir], size=256, local_mean_norm=True)
    image, grade, referable = dataset[0]
    assert len(dataset) == 16
    assert image.shape == (3, 256, 256)
    assert (grade, referable) == (int(rows["dr_grade"].iloc[0]), float(grade >= 2))
    # the grey border of local mean normalisation maps to ~0.5 before ImageNet scaling
    corner = image[:, 0, 0].numpy() * np.array([0.229, 0.224, 0.225]) + IMAGENET_MEAN
    assert corner == pytest.approx(128 / 255, abs=0.01)


def test_augment_keeps_shape(fixture_shard: tuple[Path, pd.DataFrame]) -> None:
    shard_dir, manifest = fixture_shard
    rows = select_rows(manifest, ["train"])
    dataset = FundusDataset(
        rows, [shard_dir], size=128, local_mean_norm=False, augment=train_augment()
    )
    first, second = dataset[0][0], dataset[0][0]
    assert first.shape == second.shape == (3, 128, 128)
    assert not torch.equal(first, second)  # random augmentation


def test_sampler_favours_rare_grades() -> None:
    grades = np.array([0] * 90 + [4] * 10)
    sampler = grade_balancing_sampler(grades, seed=0)
    drawn = grades[list(sampler)]
    # weight 1/sqrt(count): grade 4 share = 10/sqrt(10) / (90/sqrt(90) + 10/sqrt(10)) = 0.25
    assert (drawn == 4).mean() == pytest.approx(0.25, abs=0.1)
