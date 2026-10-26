"""Hard data invariants of the image manifest, as pure functions.

Each check returns the violations it found; an empty result means the invariant holds. The
same functions back the Dagster asset checks and the unit tests. Invariant 3 (no near-duplicate
phash across train/eval) lives with the hashing code.
"""

import pandas as pd

from retinaguard.data.schemas import EXTERNAL_SOURCES, TRAINING_SPLITS

STRATIFICATION_TOLERANCE = 0.02


def duplicate_image_ids(manifest: pd.DataFrame) -> list[str]:
    """Invariant 1: ``image_id`` is unique."""
    dupes = manifest.loc[manifest["image_id"].duplicated(), "image_id"]
    return sorted(set(dupes))


def patients_in_several_splits(manifest: pd.DataFrame) -> list[str]:
    """Invariant 2: no ``patient_id`` appears in more than one split."""
    with_patient = manifest.dropna(subset=["patient_id"])
    n_splits = with_patient.groupby("patient_id")["split"].nunique()
    return sorted(n_splits[n_splits > 1].index)


def external_images_in_training(manifest: pd.DataFrame) -> list[str]:
    """Invariant 4: external sources are never in train/val/calib."""
    leaked = manifest["source"].isin(EXTERNAL_SOURCES) & manifest["split"].isin(TRAINING_SPLITS)
    return sorted(manifest.loc[leaked, "image_id"])


def stratification_deviations(
    manifest: pd.DataFrame, tolerance: float = STRATIFICATION_TOLERANCE
) -> list[str]:
    """Invariant 5: per source, each split's grade distribution is within ``tolerance`` of the
    source's overall distribution. Returns ``"source/split/grade: split% vs source%"`` entries.
    """
    graded = manifest.dropna(subset=["dr_grade"])
    deviations = []
    for source, rows in graded.groupby("source"):
        overall = rows["dr_grade"].value_counts(normalize=True)
        per_split = pd.crosstab(rows["split"], rows["dr_grade"], normalize="index")
        for split, dist in per_split.iterrows():
            for grade, share in overall.items():
                split_share = float(dist.get(grade, 0.0))
                if abs(split_share - share) > tolerance:
                    deviations.append(f"{source}/{split}/{grade}: {split_share:.3f} vs {share:.3f}")
    return deviations
