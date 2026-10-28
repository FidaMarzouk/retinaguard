"""APTOS 2019 (Kaggle "APTOS 2019 Blindness Detection", Aravind Eye Hospital, India).

Only the labelled training set is used, as an external test set. ``train.csv`` has ``id_code``
and ``diagnosis`` (ICDR 0-4); images are ``train_images/{id_code}.png``. There are no patient or
eye identifiers, so every image is its own group.
"""

import re
from pathlib import Path

import pandas as pd

from retinaguard.data.labels import LABEL_MAP_VERSION, is_referable, parse_icdr
from retinaguard.data.schemas import LabelManifestSchema

SOURCE = "aptos"
SITE_COUNTRY = "IN"
_ID = re.compile(r"^[0-9a-f]{12}$")


class KaggleImages:
    """Raw APTOS images in a Kaggle competition mount (``train_images/*.png``)."""

    def __init__(self, data_dir: Path) -> None:
        png_dir = data_dir / "train_images"
        if not png_dir.is_dir():
            raise FileNotFoundError(f"no train_images/ folder in {data_dir}")
        self._files = {p.stem: p for p in png_dir.glob("*.png")}

    def names(self) -> set[str]:
        """Image names as in ``train.csv`` (``id_code``)."""
        return set(self._files)

    def read(self, image_id: str) -> bytes:
        """Raw bytes for a manifest ``image_id`` (``aptos:{id_code}``)."""
        return self._files[image_id.removeprefix(f"{SOURCE}:")].read_bytes()


def read_labels(labels_csv: Path) -> pd.DataFrame:
    """Read ``train.csv`` into validated manifest rows."""
    return rows_from_labels(pd.read_csv(labels_csv, dtype=str))


def rows_from_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Map the ``id_code``/``diagnosis`` columns of ``train.csv`` to manifest rows."""
    ids = labels["id_code"].str.strip()
    bad = ids[~ids.str.fullmatch(_ID)]
    if not bad.empty:
        raise ValueError(f"unexpected APTOS id codes: {bad.head().tolist()}")

    grades = [parse_icdr(raw) for raw in labels["diagnosis"]]
    missing = pd.Series(None, index=labels.index, dtype="str")
    rows = pd.DataFrame(
        {
            "image_id": SOURCE + ":" + ids,
            "source": SOURCE,
            "site_country": SITE_COUNTRY,
            "patient_id": missing,
            "eye": missing,
            "raw_path": f"{SOURCE}/train_images/" + ids + ".png",
            "dr_grade": pd.array(grades, dtype="Int64"),
            "referable": pd.array([is_referable(g) for g in grades], dtype="boolean"),
            "quality_label": missing,
            "has_lesion_mask": False,
            "label_raw": labels["diagnosis"].str.strip(),
            "label_map_version": LABEL_MAP_VERSION,
        }
    )
    return LabelManifestSchema.validate(rows)
