"""DDR grading set (Li et al., Information Sciences 501, 2019, doi:10.1016/j.ins.2019.06.011;
github.com/nkicsl/DDR-dataset, CC BY-NC-SA 4.0). Images from 147 hospitals in China.

Only ``DR_grading`` is used, as an external test set. It ships inside ``DDR-dataset.zip.001``
... ``.010`` (byte-range parts of one zip) as::

    DDR-dataset/DR_grading/{train,valid,test}/*.jpg
    DDR-dataset/DR_grading/{train,valid,test}.txt    one "<file name> <grade>" per line

Grades are ICDR 0-4 plus 5 for "ungradable". Ungradable images get no DR grade and
``quality_label`` "reject", so they can test the quality gate. Image ids carry the subset
(``ddr:valid_20170323093705822``), as the subsets are released separately. There are no patient
or eye identifiers.
"""

import re
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

from retinaguard.data.archive import open_split_zip
from retinaguard.data.labels import LABEL_MAP_VERSION, is_referable, parse_icdr
from retinaguard.data.schemas import LabelManifestSchema

SOURCE = "ddr"
SITE_COUNTRY = "CN"
SUBSETS = ("train", "valid", "test")
UNGRADABLE = "5"
GRADING_DIR = "DR_grading"
_LINE = re.compile(r"^(?P<name>\S+\.jpg)\s+(?P<grade>\d)$")


class GradingImages:
    """Images and label files of ``DR_grading``, from the split zip or an extracted folder."""

    def __init__(self, location: Path) -> None:
        """``location``: ``DDR-dataset.zip.001`` or any folder containing ``DR_grading/``."""
        self._archive: zipfile.ZipFile | None = None
        self._files: dict[str, Path | zipfile.ZipInfo] = {}
        self._labels: dict[str, Path | zipfile.ZipInfo] = {}
        if location.is_file():
            self._archive = open_split_zip(location)
            entries: list[tuple[PurePosixPath, Path | zipfile.ZipInfo]] = [
                (PurePosixPath(i.filename), i) for i in self._archive.infolist() if not i.is_dir()
            ]
        else:
            entries = [
                (PurePosixPath(p.relative_to(location).as_posix()), p)
                for p in location.rglob("*")
                if p.is_file()
            ]
        for path, entry in entries:
            if GRADING_DIR not in path.parts[:-1]:
                continue
            parent = path.parent.name
            if parent == GRADING_DIR and path.suffix == ".txt" and path.stem in SUBSETS:
                self._labels[path.stem] = entry
            elif (
                parent in SUBSETS
                and path.parent.parent.name == GRADING_DIR
                and path.suffix.lower() == ".jpg"
            ):
                self._files[image_id(parent, path.stem)] = entry
        missing = sorted(set(SUBSETS) - set(self._labels))
        if missing:
            raise FileNotFoundError(f"no {GRADING_DIR}/{{{','.join(missing)}}}.txt in {location}")

    def ids(self) -> set[str]:
        return set(self._files)

    def read(self, image_id: str) -> bytes:
        return self._read(self._files[image_id])

    def read_labels(self) -> pd.DataFrame:
        """All three label files as validated manifest rows."""
        frames = [
            parse_label_file(self._read(self._labels[subset]).decode(), subset)
            for subset in SUBSETS
        ]
        return rows_from_labels(pd.concat(frames, ignore_index=True))

    def _read(self, entry: Path | zipfile.ZipInfo) -> bytes:
        if isinstance(entry, Path):
            return entry.read_bytes()
        assert self._archive is not None
        return self._archive.read(entry)


def image_id(subset: str, stem: str) -> str:
    return f"{SOURCE}:{subset}_{stem}"


def parse_label_file(text: str, subset: str) -> pd.DataFrame:
    """``name``, ``grade``, ``subset`` from one ``{subset}.txt``; blank lines are skipped."""
    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        match = _LINE.match(line.strip())
        if match is None:
            raise ValueError(f"{subset}.txt line {number}: unexpected {line!r}")
        rows.append((match["name"], match["grade"]))
    return pd.DataFrame(rows, columns=["name", "grade"]).assign(subset=subset)


def rows_from_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Map ``name``/``grade``/``subset`` to manifest rows."""
    ungradable = labels["grade"] == UNGRADABLE
    grades = [
        None if bad else parse_icdr(g) for g, bad in zip(labels["grade"], ungradable, strict=True)
    ]
    stems = labels["name"].str.removesuffix(".jpg")
    missing = pd.Series(None, index=labels.index, dtype="str")
    rows = pd.DataFrame(
        {
            "image_id": [image_id(s, n) for s, n in zip(labels["subset"], stems, strict=True)],
            "source": SOURCE,
            "site_country": SITE_COUNTRY,
            "patient_id": missing,
            "eye": missing,
            "raw_path": f"{SOURCE}/{GRADING_DIR}/" + labels["subset"] + "/" + labels["name"],
            "dr_grade": pd.array(grades, dtype="Int64"),
            "referable": pd.array(
                [None if g is None else is_referable(g) for g in grades], dtype="boolean"
            ),
            "quality_label": missing.where(~ungradable, "reject"),
            "has_lesion_mask": False,
            "label_raw": labels["grade"],
            "label_map_version": LABEL_MAP_VERSION,
        }
    )
    return LabelManifestSchema.validate(rows)
