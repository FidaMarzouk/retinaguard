"""EyePACS (Kaggle "Diabetic Retinopathy Detection", 2015) → manifest rows.

Only the labelled training set is used. Image names are ``{patient}_{left|right}`` and the
images sit in ``train/{name}.jpeg`` once the competition archive is extracted.
"""

import re
import zipfile
from pathlib import Path

import pandas as pd

from retinaguard.data.archive import iter_members, open_split_zip
from retinaguard.data.labels import LABEL_MAP_VERSION, is_referable, parse_icdr
from retinaguard.data.schemas import LabelManifestSchema

SOURCE = "eyepacs"
SITE_COUNTRY = "US"
_NAME = re.compile(r"^(?P<patient>\d+)_(?P<eye>left|right)$")


class KaggleImages:
    """Raw EyePACS images in a Kaggle competition mount, extracted or as ``train.zip.NNN`` parts."""

    def __init__(self, data_dir: Path) -> None:
        jpeg_dir = next(
            (d for d in (data_dir / "train", data_dir / "train" / "train") if d.is_dir()), None
        )
        first_part = data_dir / "train.zip.001"
        self._archive: zipfile.ZipFile | None = None
        if jpeg_dir is not None:
            self.layout = "extracted"
            self._files: dict[str, Path | zipfile.ZipInfo] = {
                p.stem: p for p in jpeg_dir.glob("*.jpeg")
            }
        elif first_part.exists():
            self.layout = "split_zip"
            self._archive = open_split_zip(first_part)
            self._files = {Path(i.filename).stem: i for i in iter_members(self._archive, ".jpeg")}
        else:
            raise FileNotFoundError(f"no train/ folder or train.zip.001 in {data_dir}")

    def names(self) -> set[str]:
        """Image names as in ``trainLabels.csv`` (``{patient}_{eye}``)."""
        return set(self._files)

    def read(self, image_id: str) -> bytes:
        """Raw bytes for a manifest ``image_id`` (``eyepacs:{name}``)."""
        entry = self._files[image_id.removeprefix(f"{SOURCE}:")]
        if isinstance(entry, Path):
            return entry.read_bytes()
        assert self._archive is not None
        return self._archive.read(entry)


def read_labels(labels_csv: Path) -> pd.DataFrame:
    """Read ``trainLabels.csv`` (or the ``.csv.zip`` Kaggle ships) into validated manifest rows."""
    labels = pd.read_csv(labels_csv, dtype=str)
    return rows_from_labels(labels)


def rows_from_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Map the ``image``/``level`` columns of ``trainLabels.csv`` to manifest rows."""
    names = labels["image"].str.strip()
    parts = names.str.extract(_NAME)
    bad = names[parts["patient"].isna()]
    if not bad.empty:
        raise ValueError(f"unexpected EyePACS image names: {bad.head().tolist()}")

    grades = [parse_icdr(raw) for raw in labels["level"]]
    rows = pd.DataFrame(
        {
            "image_id": SOURCE + ":" + names,
            "source": SOURCE,
            "site_country": SITE_COUNTRY,
            "patient_id": SOURCE + ":" + parts["patient"],
            "eye": parts["eye"],
            "raw_path": f"{SOURCE}/train/" + names + ".jpeg",
            "dr_grade": pd.array(grades, dtype="Int64"),
            "referable": pd.array([is_referable(g) for g in grades], dtype="boolean"),
            "quality_label": pd.Series(None, index=labels.index, dtype="str"),
            "has_lesion_mask": False,
            "label_raw": labels["level"].str.strip(),
            "label_map_version": LABEL_MAP_VERSION,
        }
    )
    return LabelManifestSchema.validate(rows)
