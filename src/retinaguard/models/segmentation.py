"""Lesion segmentation model, loss and pixel metrics.

A U-Net with an ImageNet encoder predicts one sigmoid channel per lesion type (``LESIONS``).
Metrics are built from per-image counts so that sets can be pooled and bootstrapped over
images: true/false positives and false negatives at threshold 0.5 (for Dice), and histograms
of predicted probabilities over lesion and background pixels (for AUPR, the area under the
precision-recall curve, which suits tiny lesions better than Dice).
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import segmentation_models_pytorch as smp
import torch
from torch import nn

from retinaguard.data.segmentation import LESIONS

THRESHOLD = 0.5
N_BINS = 200


def seg_model(encoder: str, *, pretrained: bool) -> nn.Module:
    model: nn.Module = smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        classes=len(LESIONS),
    )
    return model


class SegLoss(nn.Module):
    """Dice plus focal loss over the lesion channels (multi-label)."""

    def __init__(self) -> None:
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="multilabel", from_logits=True)
        self.focal = smp.losses.FocalLoss(mode="multilabel")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        loss: torch.Tensor = self.dice(logits, target) + self.focal(logits, target)
        return loss


@dataclass
class PixelStats:
    """Per-image counts, arrays of shape (images, lesions[, bins])."""

    tp: npt.NDArray[np.int64]
    fp: npt.NDArray[np.int64]
    fn: npt.NDArray[np.int64]
    positive_hist: npt.NDArray[np.int64]
    negative_hist: npt.NDArray[np.int64]

    def take(self, index: npt.ArrayLike) -> "PixelStats":
        i = np.asarray(index)
        return PixelStats(
            self.tp[i], self.fp[i], self.fn[i], self.positive_hist[i], self.negative_hist[i]
        )


def image_stats(
    probs: npt.NDArray[np.floating], masks: npt.NDArray[np.bool_], n_bins: int = N_BINS
) -> PixelStats:
    """Counts of one image: ``probs`` and ``masks`` are (lesions, H, W)."""
    predicted = probs >= THRESHOLD
    flat_probs = probs.reshape(len(probs), -1)
    flat_masks = masks.reshape(len(masks), -1)
    bins = np.minimum((flat_probs * n_bins).astype(np.int64), n_bins - 1)
    positive = np.stack(
        [np.bincount(b[m], minlength=n_bins) for b, m in zip(bins, flat_masks, strict=True)]
    )
    negative = np.stack(
        [np.bincount(b[~m], minlength=n_bins) for b, m in zip(bins, flat_masks, strict=True)]
    )
    return PixelStats(
        tp=(predicted & masks).sum(axis=(1, 2))[None],
        fp=(predicted & ~masks).sum(axis=(1, 2))[None],
        fn=(~predicted & masks).sum(axis=(1, 2))[None],
        positive_hist=positive[None],
        negative_hist=negative[None],
    )


def concat_stats(parts: list[PixelStats]) -> PixelStats:
    return PixelStats(
        *(
            np.concatenate([getattr(p, f) for p in parts])
            for f in ("tp", "fp", "fn", "positive_hist", "negative_hist")
        )
    )


def dice(stats: PixelStats) -> npt.NDArray[np.float64]:
    """Dice per lesion over all pixels of the set; NaN where a lesion is absent and unpredicted."""
    tp, fp, fn = stats.tp.sum(0), stats.fp.sum(0), stats.fn.sum(0)
    denominator = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, 2 * tp / denominator, np.nan)


def aupr(stats: PixelStats) -> npt.NDArray[np.float64]:
    """Average precision per lesion over all pixels of the set, from the probability histograms
    (thresholds at the bin edges); NaN where the set has no lesion pixel."""
    positive = stats.positive_hist.sum(0)[:, ::-1].astype(np.float64)  # high scores first
    negative = stats.negative_hist.sum(0)[:, ::-1].astype(np.float64)
    tp, fp = positive.cumsum(1), negative.cumsum(1)
    total = positive.sum(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 1.0)
        recall_step = positive / total
    result = (precision * recall_step).sum(1)
    return np.where(total[:, 0] > 0, result, np.nan)
