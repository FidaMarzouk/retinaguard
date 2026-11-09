"""Training and evaluation data for the DR grader and the quality model, read from the manifest
and processed shards.

Each grader item is the stored fundus crop, optionally augmented, optionally local-mean
normalised, then scaled with ImageNet statistics: ``(image CHW float32, ICDR grade, referable
0/1)``. A quality item is ``(image, quality class)`` with the classes of ``QUALITY_CLASSES``.
"""

import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from retinaguard.data.preprocess import Image, local_mean_normalise
from retinaguard.data.processing import ProcessedImages, shard_roots
from retinaguard.models.quality import QUALITY_CLASSES

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def select_rows(
    manifest: pd.DataFrame,
    splits: Iterable[str],
    *,
    exclude_quality_reject: bool = True,
    limit_per_split: int | None = None,
) -> pd.DataFrame:
    """Processed, graded images of ``splits``; quality rejects dropped unless asked otherwise.

    ``limit_per_split`` keeps the first images by ``image_id`` of each split (smoke tests).
    """
    rows = manifest[
        manifest["split"].isin(list(splits))
        & manifest["processed_path"].notna()
        & manifest["dr_grade"].notna()
    ]
    if exclude_quality_reject:
        rows = rows[rows["quality_label"].ne("reject").fillna(True)]
    rows = rows.sort_values("image_id")
    if limit_per_split is not None:
        rows = rows.groupby("split", sort=False).head(limit_per_split)
    return rows.reset_index(drop=True)


def select_quality_rows(
    manifest: pd.DataFrame, splits: Iterable[str], *, limit_per_split: int | None = None
) -> pd.DataFrame:
    """Processed EyePACS images of ``splits`` with an EyeQ quality label."""
    rows = manifest[
        manifest["split"].isin(list(splits))
        & (manifest["source"] == "eyepacs")
        & manifest["processed_path"].notna()
        & manifest["quality_label"].notna()
    ].sort_values("image_id")
    if limit_per_split is not None:
        rows = rows.groupby("split", sort=False).head(limit_per_split)
    return rows.reset_index(drop=True)


def train_augment() -> A.Compose:
    """Geometry (fundus photos are rotation-tolerant) and mild photometric changes, plus
    occasional blur and JPEG artefacts to mimic cheaper cameras."""
    return A.Compose(
        [
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Affine(
                rotate=(-180, 180),
                scale=(0.95, 1.05),
                translate_percent=(-0.03, 0.03),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=1.0,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=0, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
            A.ImageCompression(quality_range=(40, 90), p=0.1),
        ]
    )


def to_input(img: Image, *, size: int, local_mean_norm: bool) -> torch.Tensor:
    """Resize if needed, optionally normalise the local mean, scale to ImageNet statistics."""
    if img.shape[0] != size:
        interpolation = cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_CUBIC
        img = np.asarray(cv2.resize(img, (size, size), interpolation=interpolation), np.uint8)
    if local_mean_norm:
        img = local_mean_normalise(img)
    scaled = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(scaled.transpose(2, 0, 1)))


class FundusDataset(Dataset[tuple[torch.Tensor, int, float]]):
    """Rows of ``select_rows`` read from processed shards found in ``shard_dirs``."""

    def __init__(
        self,
        rows: pd.DataFrame,
        shard_dirs: Sequence[Path],
        *,
        size: int,
        local_mean_norm: bool,
        augment: A.Compose | None = None,
    ) -> None:
        self.image_ids: list[str] = rows["image_id"].tolist()
        # Ungraded images (DDR "ungradable") get grade -1; they are predicted, never trained on.
        self.grades = rows["dr_grade"].astype("Int64").fillna(-1).astype(int).to_numpy()
        self.referable = (self.grades >= 2).astype(np.float32)
        self._paths = dict(zip(rows["image_id"], rows["processed_path"], strict=True))
        self._roots = shard_roots(*shard_dirs)
        self._size = size
        self._local_mean_norm = local_mean_norm
        self._augment = augment
        self._reader: ProcessedImages | None = None
        self._reader_pid = -1

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, float]:
        img = self._read(self.image_ids[index])
        if self._augment is not None:
            img = self._augment(image=img)["image"]
        image = to_input(img, size=self._size, local_mean_norm=self._local_mean_norm)
        return image, int(self.grades[index]), float(self.referable[index])

    def _read(self, image_id: str) -> Image:
        # One reader per process: zip handles must not be shared with forked loader workers.
        if self._reader is None or self._reader_pid != os.getpid():
            self._reader = ProcessedImages(self._paths, self._roots)
            self._reader_pid = os.getpid()
        return self._reader.read(image_id)


class QualityDataset(FundusDataset):
    """Rows of ``select_quality_rows`` as ``(image, quality class)``; rows without a quality
    label (images only predicted) get class -1."""

    def __init__(
        self,
        rows: pd.DataFrame,
        shard_dirs: Sequence[Path],
        *,
        size: int,
        augment: A.Compose | None = None,
    ) -> None:
        super().__init__(
            rows.assign(dr_grade=None),
            shard_dirs,
            size=size,
            local_mean_norm=False,
            augment=augment,
        )
        codes = {name: code for code, name in enumerate(QUALITY_CLASSES)}
        self.labels = rows["quality_label"].map(codes).fillna(-1).astype(int).to_numpy()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, float]:
        image, _, _ = super().__getitem__(index)
        return image, int(self.labels[index]), 0.0


def quality_augment() -> A.Compose:
    """Geometry only: blur, exposure and compression changes are what the quality labels
    describe, so photometric augmentation would corrupt them."""
    return A.Compose(
        [
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Affine(
                rotate=(-180, 180),
                scale=(0.95, 1.05),
                translate_percent=(-0.03, 0.03),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=1.0,
            ),
        ]
    )


def grade_balancing_sampler(grades: np.ndarray, seed: int) -> WeightedRandomSampler:
    """Samples images with weight 1 / sqrt(grade frequency): rare grades are seen more often
    without repeating the few grade 3-4 images as much as full balancing would."""
    counts = np.bincount(grades, minlength=5).astype(np.float64)
    weights = 1.0 / np.sqrt(counts[grades])
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights.tolist(), num_samples=len(grades), replacement=True, generator=generator
    )
