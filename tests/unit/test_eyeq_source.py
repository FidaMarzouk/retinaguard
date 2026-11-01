from pathlib import Path

import pandas as pd
import pytest

from retinaguard.data.sources import eyepacs, eyeq


def _labels(images: list[str], quality: list[str], grades: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"image": images, "quality": quality, "DR_grade": grades, "eyeq_subset": "train"},
        dtype=str,
    )


def _manifest() -> pd.DataFrame:
    return eyepacs.rows_from_labels(
        pd.DataFrame({"image": ["1_left", "1_right", "2_left"], "level": ["0", "2", "4"]})
    )


def test_rows_from_labels() -> None:
    rows = eyeq.rows_from_labels(_labels(["1_left.jpeg", "2_left.jpeg"], ["0", "2"], ["0", "4"]))
    assert rows["image_id"].tolist() == ["eyepacs:1_left", "eyepacs:2_left"]
    assert rows["quality_label"].tolist() == ["good", "reject"]


def test_rejects_unknown_quality() -> None:
    with pytest.raises(ValueError, match="unexpected EyeQ quality"):
        eyeq.rows_from_labels(_labels(["1_left.jpeg"], ["3"], ["0"]))


def test_rejects_repeated_images() -> None:
    with pytest.raises(ValueError, match="repeat"):
        eyeq.rows_from_labels(_labels(["1_left.jpeg"] * 2, ["0", "1"], ["0", "0"]))


def test_attach_quality() -> None:
    labels = eyeq.rows_from_labels(
        _labels(["1_left.jpeg", "2_left.jpeg", "9_left.jpeg"], ["1", "2", "0"], ["0", "4", "1"])
    )
    manifest = eyeq.attach_quality(_manifest(), labels)
    quality = manifest["quality_label"]
    assert quality[[0, 2]].tolist() == ["usable", "reject"]
    assert pd.isna(quality[1])


def test_attach_quality_rejects_grade_mismatch() -> None:
    labels = eyeq.rows_from_labels(_labels(["1_right.jpeg"], ["0"], ["3"]))
    with pytest.raises(ValueError, match="grades differ"):
        eyeq.attach_quality(_manifest(), labels)


def test_read_labels(tmp_path: Path) -> None:
    for name, image in zip(eyeq.LABEL_FILES, ["1_left.jpeg", "7_right.jpeg"], strict=True):
        (tmp_path / name).write_text(f",image,quality,DR_grade\n0,{image},1,0\n")
    labels = eyeq.read_labels(tmp_path)
    assert labels["eyeq_subset"].tolist() == ["train", "test"]
