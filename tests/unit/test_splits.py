import numpy as np
import pandas as pd
import pytest

from retinaguard.data.invariants import (
    duplicates_across_splits,
    patients_in_several_splits,
    stratification_deviations,
)
from retinaguard.data.splits import SPLIT_FRACTIONS, assign_splits, split_groups


def _manifest(n_patients: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Two eyes per patient, grades drawn with an EyePACS-like imbalance."""
    rng = np.random.default_rng(seed)
    grades = rng.choice(5, size=2 * n_patients, p=[0.73, 0.07, 0.15, 0.03, 0.02])
    patients = np.repeat(np.arange(n_patients), 2)
    eyes = np.tile(["left", "right"], n_patients)
    return pd.DataFrame(
        {
            "image_id": [f"eyepacs:{p}_{e}" for p, e in zip(patients, eyes, strict=True)],
            "source": "eyepacs",
            "patient_id": [f"eyepacs:{p}" for p in patients],
            "dr_grade": pd.array(grades, dtype="Int64"),
        }
    )


def _with_splits(manifest: pd.DataFrame, pairs: tuple[tuple[str, str], ...] = ()) -> pd.DataFrame:
    return manifest.assign(split=assign_splits(manifest, pairs)["split"].to_numpy())


def test_split_fractions_and_invariants() -> None:
    m = _with_splits(_manifest())
    shares = m["split"].value_counts(normalize=True)
    for name, fraction in SPLIT_FRACTIONS.items():
        assert shares[name] == pytest.approx(fraction, abs=0.01)
    assert patients_in_several_splits(m) == []
    assert stratification_deviations(m) == []


def test_splits_are_reproducible_and_seeded() -> None:
    m = _manifest()
    first = assign_splits(m)
    assert first.equals(assign_splits(m))
    assert first.equals(assign_splits(m.sample(frac=1, random_state=1)).loc[m.index])
    assert not first["split"].equals(assign_splits(m, seed=1)["split"])


def test_duplicate_pairs_merge_patients_into_one_split() -> None:
    m = _manifest(n_patients=200)
    pairs = [(f"eyepacs:{i}_left", f"eyepacs:{i + 1}_right") for i in range(0, 40, 2)]
    pairs.append(("eyepacs:1_left", "eyepacs:2_left"))  # chains 0-1-2-3 into one group
    groups = split_groups(m, pairs)
    assert groups[m["patient_id"].isin([f"eyepacs:{i}" for i in range(4)])].nunique() == 1
    assert groups.iloc[0] == "eyepacs:0"
    m = _with_splits(m, tuple(pairs))
    assert duplicates_across_splits(m, pairs) == []


def test_unknown_patient_is_its_own_group() -> None:
    m = _manifest(n_patients=3)
    m["patient_id"] = [None] * 2 + list(m["patient_id"].iloc[2:])
    groups = split_groups(m, [])
    assert groups.iloc[0] == "eyepacs:0_left"
    assert groups.iloc[1] == "eyepacs:0_right"


def test_pairs_with_unknown_images_are_ignored() -> None:
    m = _manifest(n_patients=3)
    assert split_groups(m, [("eyepacs:0_left", "aptos:x")]).nunique() == 3


def test_fractions_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        assign_splits(_manifest(n_patients=3), fractions={"train": 0.7, "val": 0.2})


def test_duplicates_across_splits_reports_pairs() -> None:
    m = pd.DataFrame({"image_id": ["a", "b", "c"], "split": ["train", "val", "train"]})
    pairs = [("a", "b"), ("a", "c"), ("a", "missing")]
    assert duplicates_across_splits(m, pairs) == ["a | b: train vs val"]
