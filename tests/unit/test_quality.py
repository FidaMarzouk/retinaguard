import numpy as np
import pandas as pd
import pytest
import torch

from retinaguard.config import load_config
from retinaguard.data.datasets import QualityDataset, select_quality_rows
from retinaguard.models.quality import (
    QUALITY_COLUMNS,
    class_weights,
    confusion,
    quality_metrics,
    quality_model,
)
from retinaguard.train.evaluate_quality import evaluate_quality


def test_quality_model_outputs_three_classes() -> None:
    model = quality_model("test_efficientnet", pretrained=False)
    assert model(torch.zeros(2, 3, 64, 64)).shape == (2, 3)


def test_class_weights_balance_classes() -> None:
    labels = np.array([0] * 6 + [1] * 3 + [2] * 1)
    weights = class_weights(labels)
    assert np.allclose(np.bincount(labels) * weights, len(labels) / 3)
    assert weights[labels].mean() == pytest.approx(1)
    with pytest.raises(ValueError, match="every quality class"):
        class_weights([0, 1, 1])


def test_quality_metrics_and_confusion() -> None:
    labels = np.array([0, 0, 1, 2, 2])
    probs = np.eye(3)[[0, 1, 1, 2, 0]]
    result = quality_metrics(labels, probs)
    assert result["accuracy"] == pytest.approx(3 / 5)
    assert result["recall_good"] == pytest.approx(0.5)
    assert result["recall_reject"] == pytest.approx(0.5)
    table = confusion(labels, probs)
    assert table.loc["reject", "good"] == 1
    assert table.to_numpy().sum() == 5


def _manifest() -> pd.DataFrame:
    rows = [
        # image_id, source, split, quality, processed
        ("eyepacs:1", "eyepacs", "train", "good", "p"),
        ("eyepacs:2", "eyepacs", "train", None, "p"),  # no EyeQ label
        ("eyepacs:3", "eyepacs", "train", "reject", None),  # crop failed
        ("eyepacs:4", "eyepacs", "test_internal", "usable", "p"),
        ("eyepacs:5", "eyepacs", "test_internal", "reject", "p"),
        ("eyepacs:6", "eyepacs", "test_internal", "good", "p"),
        ("ddr:a", "ddr", "test_external", "reject", "p"),  # ungradable
        ("ddr:b", "ddr", "test_external", None, "p"),
        ("ddr:c", "ddr", "test_external", None, "p"),
        ("aptos:a", "aptos", "test_external", None, "p"),
    ]
    frame = pd.DataFrame(
        rows, columns=["image_id", "source", "split", "quality_label", "processed_path"]
    )
    return frame.assign(split_group=frame["image_id"], dr_grade=pd.array([0] * 10, "Int64"))


def test_select_quality_rows_and_dataset_labels() -> None:
    manifest = _manifest()
    assert select_quality_rows(manifest, ["train"])["image_id"].tolist() == ["eyepacs:1"]
    rows = manifest[manifest["source"] == "ddr"]
    dataset = QualityDataset(rows, [], size=64)
    assert dataset.labels.tolist() == [2, -1, -1]


def test_evaluate_quality_sets() -> None:
    manifest = _manifest()
    predicted = {"eyepacs:4": 1, "eyepacs:5": 2, "eyepacs:6": 2, "ddr:a": 2, "ddr:b": 2}
    ids = manifest["image_id"][manifest["split"] != "train"]
    probs = np.eye(3)[[predicted.get(i, 0) for i in ids]]
    predictions = pd.DataFrame(probs, columns=QUALITY_COLUMNS).assign(image_id=ids.to_numpy())
    table = evaluate_quality(manifest, predictions, n_resamples=20)
    assert set(table.index) == {"eyepacs_test", "ddr", "aptos"}
    assert table.loc["eyepacs_test", "recall_reject"] == 1.0
    assert table.loc["eyepacs_test", "recall_good"] == 0.0
    assert table.loc["ddr", "recall_reject"] == 1.0
    assert table.loc["ddr", "reject_rate_graded"] == pytest.approx(0.5)
    assert table.loc["aptos", "reject_rate"] == 0.0


def test_quality_config_composes() -> None:
    cfg = load_config("train/quality", ["train.epochs=3"])
    assert cfg.run_name == "quality-efficientnet_b0"
    assert cfg.data.image_size == 512
