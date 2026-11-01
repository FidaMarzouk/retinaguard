import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from retinaguard.data.sources import ddr

LABELS = {
    "train": "20170001.jpg 0\n20170002.jpg 5\n",
    "valid": "20170003.jpg 2\n\n",
    "test": "20170004.jpg 4\n",
}


def _members() -> dict[str, bytes]:
    members = {f"DDR-dataset/DR_grading/{s}.txt": text.encode() for s, text in LABELS.items()}
    for subset, text in LABELS.items():
        for line in text.split():
            if line.endswith(".jpg"):
                members[f"DDR-dataset/DR_grading/{subset}/{line}"] = f"{subset}/{line}".encode()
    members["DDR-dataset/lesion_segmentation/train/image/x.jpg"] = b"not grading"
    return members


def test_rows_from_labels() -> None:
    labels = pd.concat(ddr.parse_label_file(t, s) for s, t in LABELS.items())
    rows = ddr.rows_from_labels(labels.reset_index(drop=True))

    assert rows["image_id"].tolist() == [
        "ddr:train_20170001",
        "ddr:train_20170002",
        "ddr:valid_20170003",
        "ddr:test_20170004",
    ]
    assert rows["raw_path"].iloc[2] == "ddr/DR_grading/valid/20170003.jpg"
    assert rows["dr_grade"].tolist() == [0, pd.NA, 2, 4]
    assert rows["referable"].tolist() == [False, pd.NA, True, True]
    # "ungradable" becomes a quality reject without a DR grade
    assert rows["quality_label"].isna().tolist() == [True, False, True, True]
    assert rows["quality_label"].iloc[1] == "reject"
    assert rows["label_raw"].iloc[1] == "5"
    assert (rows["site_country"] == "CN").all()


@pytest.mark.parametrize("line", ["20170001.jpg", "20170001.png 0", "20170001.jpg 7x"])
def test_rejects_unexpected_lines(line: str) -> None:
    with pytest.raises(ValueError, match="line 1"):
        ddr.parse_label_file(line + "\n", "train")


def test_rejects_grade_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        ddr.rows_from_labels(ddr.parse_label_file("20170001.jpg 6\n", "train"))


def _check(images: ddr.GradingImages) -> None:
    rows = images.read_labels()
    assert images.ids() == set(rows["image_id"])
    assert images.read("ddr:valid_20170003") == b"valid/20170003.jpg"


def test_split_zip(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in _members().items():
            archive.writestr(name, data)
    raw = buffer.getvalue()
    half = len(raw) // 2
    (tmp_path / "DDR-dataset.zip.001").write_bytes(raw[:half])
    (tmp_path / "DDR-dataset.zip.002").write_bytes(raw[half:])

    _check(ddr.GradingImages(tmp_path / "DDR-dataset.zip.001"))


def test_extracted_folder(tmp_path: Path) -> None:
    for name, data in _members().items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(data)

    _check(ddr.GradingImages(tmp_path))


def test_requires_label_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="DR_grading"):
        ddr.GradingImages(tmp_path)
