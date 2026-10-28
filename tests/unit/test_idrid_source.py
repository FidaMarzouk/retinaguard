from pathlib import Path

import pandas as pd
import pytest

from retinaguard.data.sources import idrid


def _labels(names: list[str], grades: list[str], subset: str = "training") -> pd.DataFrame:
    return pd.DataFrame(
        {"Image name": names, "Retinopathy grade": grades, "Risk of macular edema ": "0"},
        dtype=str,
    ).assign(subset=subset)


def _archive(tmp_path: Path) -> Path:
    """A miniature ``B. Disease Grading`` folder, label files as shipped (extra empty columns)."""
    root = tmp_path / "download" / idrid.ROOT_NAME
    for subset, (folder, suffix) in idrid.SUBSETS.items():
        (root / "1. Original Images" / folder).mkdir(parents=True)
        (root / "1. Original Images" / folder / "IDRiD_001.jpg").write_bytes(subset.encode())
        labels = root / "2. Groundtruths" / f"x. IDRiD_Disease Grading_{suffix}"
        labels.parent.mkdir(exist_ok=True)
        grade = "3" if subset == "training" else "1"
        labels.write_text(
            f"Image name,Retinopathy grade,Risk of macular edema ,,\nIDRiD_001,{grade},1,,\n,,,,\n"
        )
    return root


def test_rows_from_labels() -> None:
    rows = idrid.rows_from_labels(
        pd.concat([_labels(["IDRiD_001"], ["2"]), _labels(["IDRiD_001"], [" 0"], "testing")])
    )

    assert rows["image_id"].tolist() == ["idrid:training_IDRiD_001", "idrid:testing_IDRiD_001"]
    assert rows["raw_path"].tolist() == [
        "idrid/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_001.jpg",
        "idrid/B. Disease Grading/1. Original Images/b. Testing Set/IDRiD_001.jpg",
    ]
    assert rows["dr_grade"].tolist() == [2, 0]
    assert rows["referable"].tolist() == [True, False]
    assert rows["patient_id"].isna().all()
    assert (rows["site_country"] == "IN").all()


@pytest.mark.parametrize("name", ["IDRiD_01", "IDRiD_001.jpg", "idrid_001"])
def test_rejects_unexpected_names(name: str) -> None:
    with pytest.raises(ValueError, match="unexpected IDRiD image names"):
        idrid.rows_from_labels(_labels([name], ["0"]))


def test_read_archive(tmp_path: Path) -> None:
    _archive(tmp_path)
    root = idrid.find_root(tmp_path)
    assert root.name == idrid.ROOT_NAME

    rows = idrid.read_labels(root)
    assert rows["dr_grade"].tolist() == [3, 1]

    images = idrid.GradingImages(root)
    assert images.ids() == set(rows["image_id"])
    assert images.read("idrid:testing_IDRiD_001") == b"testing"


def test_find_root_requires_layout(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"B. Disease Grading"):
        idrid.find_root(tmp_path)
