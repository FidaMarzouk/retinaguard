from pathlib import Path

import pandas as pd
import pandera.errors
import pytest

from retinaguard.data.sources import aptos


def _labels(ids: list[str], diagnoses: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"id_code": ids, "diagnosis": diagnoses}, dtype=str)


def test_rows_from_labels() -> None:
    rows = aptos.rows_from_labels(_labels(["000c1434d8d7", "001639a390f0"], ["2", "4"]))

    assert rows["image_id"].tolist() == ["aptos:000c1434d8d7", "aptos:001639a390f0"]
    assert rows["raw_path"].iloc[0] == "aptos/train_images/000c1434d8d7.png"
    assert rows["patient_id"].isna().all()
    assert rows["eye"].isna().all()
    assert rows["dr_grade"].tolist() == [2, 4]
    assert rows["referable"].tolist() == [True, True]
    assert (rows["site_country"] == "IN").all()


@pytest.mark.parametrize("id_code", ["000c1434d8d", "000C1434D8D7", "000c1434d8d7.png"])
def test_rejects_unexpected_ids(id_code: str) -> None:
    with pytest.raises(ValueError, match="unexpected APTOS id codes"):
        aptos.rows_from_labels(_labels([id_code], ["0"]))


def test_rejects_duplicate_ids() -> None:
    with pytest.raises(pandera.errors.SchemaError):
        aptos.rows_from_labels(_labels(["000c1434d8d7"] * 2, ["0", "1"]))


def test_read_labels(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    _labels(["000c1434d8d7"], ["3"]).to_csv(path, index=False)
    assert aptos.read_labels(path)["dr_grade"].tolist() == [3]


def test_kaggle_images(tmp_path: Path) -> None:
    (tmp_path / "train_images").mkdir()
    (tmp_path / "train_images" / "000c1434d8d7.png").write_bytes(b"png")

    images = aptos.KaggleImages(tmp_path)

    assert images.names() == {"000c1434d8d7"}
    assert images.read("aptos:000c1434d8d7") == b"png"


def test_kaggle_images_requires_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        aptos.KaggleImages(tmp_path)
