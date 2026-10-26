import io
import zipfile
from pathlib import Path

import pandas as pd
import pandera.errors
import pytest

from retinaguard.data.labels import LABEL_MAP_VERSION
from retinaguard.data.sources import eyepacs


def _labels(images: list[str], levels: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"image": images, "level": levels}, dtype=str)


def test_rows_from_labels() -> None:
    rows = eyepacs.rows_from_labels(_labels(["10_left", "10_right", "13_left"], ["0", "2", "4"]))

    assert rows["image_id"].tolist() == ["eyepacs:10_left", "eyepacs:10_right", "eyepacs:13_left"]
    assert rows["patient_id"].tolist() == ["eyepacs:10", "eyepacs:10", "eyepacs:13"]
    assert rows["eye"].tolist() == ["left", "right", "left"]
    assert rows["raw_path"].iloc[0] == "eyepacs/train/10_left.jpeg"
    assert rows["dr_grade"].tolist() == [0, 2, 4]
    assert rows["referable"].tolist() == [False, True, True]
    assert rows["quality_label"].isna().all()
    assert (rows["site_country"] == "US").all()
    assert (rows["label_map_version"] == LABEL_MAP_VERSION).all()


@pytest.mark.parametrize("name", ["10-left", "abc_left", "10_both", "10_left.jpeg"])
def test_rejects_unexpected_names(name: str) -> None:
    with pytest.raises(ValueError, match="unexpected EyePACS image names"):
        eyepacs.rows_from_labels(_labels([name], ["0"]))


def test_rejects_out_of_range_grade() -> None:
    with pytest.raises(ValueError, match="out of range"):
        eyepacs.rows_from_labels(_labels(["10_left"], ["7"]))


def test_rejects_duplicate_images() -> None:
    with pytest.raises(pandera.errors.SchemaError):
        eyepacs.rows_from_labels(_labels(["10_left", "10_left"], ["0", "1"]))


@pytest.mark.parametrize("suffix", [".csv", ".csv.zip"])
def test_read_labels_plain_and_zipped(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"trainLabels{suffix}"
    _labels(["10_left", "10_right"], ["1", "3"]).to_csv(path, index=False)

    rows = eyepacs.read_labels(path)

    assert rows["dr_grade"].tolist() == [1, 3]


def _write_split_zip(directory: Path, members: dict[str, bytes], part_size: int) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    data = buffer.getvalue()
    for i, start in enumerate(range(0, len(data), part_size), start=1):
        (directory / f"train.zip.{i:03d}").write_bytes(data[start : start + part_size])


def test_kaggle_images_split_zip(tmp_path: Path) -> None:
    members = {"train/10_left.jpeg": b"a" * 300, "train/10_right.jpeg": b"b" * 300}
    _write_split_zip(tmp_path, members, part_size=200)

    images = eyepacs.KaggleImages(tmp_path)

    assert images.layout == "split_zip"
    assert images.names() == {"10_left", "10_right"}
    assert images.read("eyepacs:10_right") == b"b" * 300


def test_kaggle_images_extracted(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "13_left.jpeg").write_bytes(b"jpeg")

    images = eyepacs.KaggleImages(tmp_path)

    assert images.layout == "extracted"
    assert images.read("eyepacs:13_left") == b"jpeg"


def test_kaggle_images_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        eyepacs.KaggleImages(tmp_path)
