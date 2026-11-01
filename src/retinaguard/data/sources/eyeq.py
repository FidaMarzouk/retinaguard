"""EyeQ image quality labels for EyePACS images (Fu et al., MICCAI 2019, github.com/HzFu/EyeQ).

``Label_EyeQ_train.csv`` and ``Label_EyeQ_test.csv`` have ``image`` (EyePACS file name, e.g.
``10009_left.jpeg``), ``quality`` (0 good, 1 usable, 2 reject) and ``DR_grade``. EyeQ adds no
images: its labels attach to EyePACS manifest rows. The EyeQ training images come from the
EyePACS training set; its test images come from the EyePACS competition test set.
"""

from pathlib import Path

import pandas as pd

from retinaguard.data.sources.eyepacs import SOURCE as EYEPACS

LABEL_FILES = ("Label_EyeQ_train.csv", "Label_EyeQ_test.csv")
QUALITY = {"0": "good", "1": "usable", "2": "reject"}


def read_labels(label_dir: Path) -> pd.DataFrame:
    """Both EyeQ label files as ``image_id``, ``quality_label``, ``dr_grade``, ``eyeq_subset``."""
    frames = []
    for name in LABEL_FILES:
        subset = name.removeprefix("Label_EyeQ_").removesuffix(".csv")
        frames.append(pd.read_csv(label_dir / name, dtype=str).assign(eyeq_subset=subset))
    return rows_from_labels(pd.concat(frames, ignore_index=True))


def rows_from_labels(labels: pd.DataFrame) -> pd.DataFrame:
    quality = labels["quality"].str.strip()
    unknown = quality[~quality.isin(list(QUALITY))]
    if not unknown.empty:
        raise ValueError(f"unexpected EyeQ quality values: {sorted(set(unknown))}")
    rows = pd.DataFrame(
        {
            "image_id": EYEPACS + ":" + labels["image"].str.strip().str.removesuffix(".jpeg"),
            "quality_label": quality.map(QUALITY),
            "dr_grade": labels["DR_grade"].astype(int),
            "eyeq_subset": labels["eyeq_subset"],
        }
    )
    dupes = rows["image_id"][rows["image_id"].duplicated()]
    if not dupes.empty:
        raise ValueError(f"EyeQ labels repeat images: {dupes.head().tolist()}")
    return rows


def attach_quality(manifest: pd.DataFrame, eyeq: pd.DataFrame) -> pd.DataFrame:
    """Fill ``quality_label`` for manifest images that EyeQ labels; others stay null.

    EyeQ copies the EyePACS DR grade; a different grade means the files were mismatched, so it
    raises rather than attach quality labels to the wrong images.
    """
    labels = eyeq.set_index("image_id")
    matched = manifest["image_id"].isin(labels.index)
    ids = manifest.loc[matched, "image_id"]
    differs = manifest.loc[matched, "dr_grade"].to_numpy() != labels.loc[ids, "dr_grade"].to_numpy()
    if differs.any():
        raise ValueError(f"EyeQ and manifest grades differ for {ids[differs].head().tolist()}")
    quality = manifest["quality_label"].astype(object).copy()
    quality[matched] = labels.loc[ids, "quality_label"].to_numpy()
    return manifest.assign(quality_label=quality)
