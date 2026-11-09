"""Image quality model: a timm classifier over the EyeQ classes good / usable / reject."""

import numpy as np
import numpy.typing as npt
import pandas as pd
import timm
from sklearn.metrics import f1_score, recall_score
from torch import nn

QUALITY_CLASSES = ("good", "usable", "reject")
QUALITY_COLUMNS = [f"p_{name}" for name in QUALITY_CLASSES]
REJECT = QUALITY_CLASSES.index("reject")


def quality_model(timm_name: str, *, pretrained: bool, drop_path_rate: float = 0.0) -> nn.Module:
    model: nn.Module = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=len(QUALITY_CLASSES),
        drop_path_rate=drop_path_rate,
    )
    return model


def class_weights(labels: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Inverse class frequency, scaled so the training images have mean weight 1."""
    counts = np.bincount(np.asarray(labels), minlength=len(QUALITY_CLASSES)).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(f"every quality class needs training images, got counts {counts}")
    weights = counts.sum() / (len(counts) * counts)
    return np.asarray(weights, dtype=np.float32)


def macro_f1(labels: npt.ArrayLike, predicted: npt.ArrayLike) -> float:
    return float(
        f1_score(
            labels, predicted, labels=range(len(QUALITY_CLASSES)), average="macro", zero_division=0
        )
    )


def quality_metrics(labels: npt.ArrayLike, probs: npt.ArrayLike) -> dict[str, float | int]:
    """Macro-F1, accuracy and recall of each class, from class probabilities (N, 3)."""
    y = np.asarray(labels)
    predicted = np.asarray(probs).argmax(1)
    recalls = recall_score(
        y, predicted, labels=range(len(QUALITY_CLASSES)), average=None, zero_division=np.nan
    )
    result: dict[str, float | int] = {
        "images": len(y),
        "macro_f1": macro_f1(y, predicted),
        "accuracy": float(np.mean(predicted == y)),
    }
    result |= {f"recall_{name}": float(r) for name, r in zip(QUALITY_CLASSES, recalls, strict=True)}
    return result


def confusion(labels: npt.ArrayLike, probs: npt.ArrayLike) -> pd.DataFrame:
    """True class (rows) against predicted class (columns)."""
    names = list(QUALITY_CLASSES)
    true = pd.Series(np.asarray(names)[np.asarray(labels, dtype=int)], name="true")
    predicted = pd.Series(np.asarray(names)[np.asarray(probs).argmax(1)], name="predicted")
    return pd.crosstab(true, predicted).reindex(index=names, columns=names, fill_value=0)
