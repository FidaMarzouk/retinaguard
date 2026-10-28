"""IDRiD disease grading set (Indian Diabetic Retinopathy Image Dataset, Nanded, India).

Porwal et al., IEEE DataPort, https://doi.org/10.21227/H25W98, CC BY 4.0. Only part
"B. Disease Grading" is used here, as an external test set: 413 training and 103 testing images
with an ICDR 0-4 grade (``Retinopathy grade``) and a macular edema risk, which is not used.

The archive layout is::

    B. Disease Grading/
        1. Original Images/a. Training Set/IDRiD_001.jpg ...
        1. Original Images/b. Testing Set/IDRiD_001.jpg ...
        2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv
        2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv

Both subsets number their images from ``IDRiD_001``, so the subset is part of the image id:
``idrid:training_IDRiD_001``. There are no patient or eye identifiers.
"""

import re
from pathlib import Path

import pandas as pd

from retinaguard.data.labels import LABEL_MAP_VERSION, is_referable, parse_icdr
from retinaguard.data.schemas import LabelManifestSchema

SOURCE = "idrid"
SITE_COUNTRY = "IN"
ROOT_NAME = "B. Disease Grading"
# subset -> (image folder, label file suffix)
SUBSETS = {
    "training": ("a. Training Set", "Training Labels.csv"),
    "testing": ("b. Testing Set", "Testing Labels.csv"),
}
_IMAGES = "1. Original Images"
_LABELS = "2. Groundtruths"
_NAME = re.compile(r"^IDRiD_\d{3}$")


def find_root(search_dir: Path) -> Path:
    """The ``B. Disease Grading`` folder at or below ``search_dir``."""
    found = sorted(p.parent for p in search_dir.rglob(_LABELS) if (p.parent / _IMAGES).is_dir())
    if not found:
        raise FileNotFoundError(f"no '{ROOT_NAME}' folder ({_IMAGES}/, {_LABELS}/) in {search_dir}")
    return found[0]


class GradingImages:
    """Raw images of the grading set, keyed by manifest ``image_id``."""

    def __init__(self, root: Path) -> None:
        self._files: dict[str, Path] = {}
        for subset, (folder, _) in SUBSETS.items():
            image_dir = root / _IMAGES / folder
            if not image_dir.is_dir():
                raise FileNotFoundError(f"missing image folder {image_dir}")
            for path in image_dir.iterdir():
                if path.suffix.lower() == ".jpg":
                    self._files[image_id(subset, path.stem)] = path

    def ids(self) -> set[str]:
        return set(self._files)

    def read(self, image_id: str) -> bytes:
        return self._files[image_id].read_bytes()


def image_id(subset: str, name: str) -> str:
    return f"{SOURCE}:{subset}_{name}"


def read_labels(root: Path) -> pd.DataFrame:
    """Both label files of a ``B. Disease Grading`` folder as validated manifest rows."""
    frames = []
    for subset, (_, suffix) in SUBSETS.items():
        matches = sorted((root / _LABELS).glob(f"*{suffix}"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one '*{suffix}' in {root / _LABELS}, found {matches}"
            )
        frames.append(pd.read_csv(matches[0], dtype=str).assign(subset=subset))
    return rows_from_labels(pd.concat(frames, ignore_index=True))


def rows_from_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Map ``Image name``/``Retinopathy grade`` (plus ``subset``) to manifest rows.

    The label files carry trailing empty columns and rows, which are dropped.
    """
    labels = labels.rename(columns=str.strip)
    labels = labels[labels["Image name"].notna()]
    names = labels["Image name"].str.strip()
    bad = names[~names.str.fullmatch(_NAME)]
    if not bad.empty:
        raise ValueError(f"unexpected IDRiD image names: {bad.head().tolist()}")

    raw = labels["Retinopathy grade"].str.strip()
    grades = [parse_icdr(value) for value in raw]
    folders = labels["subset"].map(lambda subset: SUBSETS[subset][0])
    missing = pd.Series(None, index=labels.index, dtype="str")
    rows = pd.DataFrame(
        {
            "image_id": [image_id(s, n) for s, n in zip(labels["subset"], names, strict=True)],
            "source": SOURCE,
            "site_country": SITE_COUNTRY,
            "patient_id": missing,
            "eye": missing,
            "raw_path": f"{SOURCE}/{ROOT_NAME}/{_IMAGES}/" + folders + "/" + names + ".jpg",
            "dr_grade": pd.array(grades, dtype="Int64"),
            "referable": pd.array([is_referable(g) for g in grades], dtype="boolean"),
            "quality_label": missing,
            "has_lesion_mask": False,
            "label_raw": raw,
            "label_map_version": LABEL_MAP_VERSION,
        }
    ).reset_index(drop=True)
    return LabelManifestSchema.validate(rows)
