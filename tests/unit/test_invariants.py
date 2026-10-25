import pandas as pd

from retinaguard.data.invariants import (
    duplicate_image_ids,
    external_images_in_training,
    patients_in_several_splits,
    stratification_deviations,
)


def _manifest(rows: list[tuple[str, str, str | None, str, int | None]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["image_id", "source", "patient_id", "split", "dr_grade"]
    ).astype({"dr_grade": "Int64"})


def test_duplicate_image_ids() -> None:
    m = _manifest(
        [
            ("eyepacs:1_left", "eyepacs", "eyepacs:1", "train", 0),
            ("eyepacs:1_left", "eyepacs", "eyepacs:1", "train", 0),
            ("eyepacs:2_left", "eyepacs", "eyepacs:2", "val", 0),
        ]
    )
    assert duplicate_image_ids(m) == ["eyepacs:1_left"]
    assert duplicate_image_ids(m.drop_duplicates()) == []


def test_patient_in_two_splits_detected() -> None:
    m = _manifest(
        [
            ("eyepacs:1_left", "eyepacs", "eyepacs:1", "train", 0),
            ("eyepacs:1_right", "eyepacs", "eyepacs:1", "test_internal", 0),
            ("eyepacs:2_left", "eyepacs", "eyepacs:2", "val", 0),
            ("eyepacs:2_right", "eyepacs", "eyepacs:2", "val", 0),
        ]
    )
    assert patients_in_several_splits(m) == ["eyepacs:1"]


def test_unknown_patients_never_conflict() -> None:
    m = _manifest(
        [
            ("aptos:a", "aptos", None, "test_external", 0),
            ("aptos:b", "aptos", None, "test_external", 1),
        ]
    )
    assert patients_in_several_splits(m) == []


def test_external_source_in_training_split_detected() -> None:
    m = _manifest(
        [
            ("aptos:a", "aptos", None, "calib", 0),
            ("idrid:b", "idrid", None, "test_external", 0),
            ("eyepacs:1_left", "eyepacs", "eyepacs:1", "train", 0),
        ]
    )
    assert external_images_in_training(m) == ["aptos:a"]


def _split_with_grades(split: str, grades: list[int]) -> list[tuple[str, str, str, str, int]]:
    return [
        (f"eyepacs:{split}{i}_left", "eyepacs", f"eyepacs:{split}{i}", split, g)
        for i, g in enumerate(grades)
    ]


def test_stratification_within_tolerance() -> None:
    grades = [0] * 80 + [2] * 20
    m = _manifest(_split_with_grades("train", grades) + _split_with_grades("val", grades))
    assert stratification_deviations(m) == []


def test_stratification_deviation_reported() -> None:
    m = _manifest(
        _split_with_grades("train", [0] * 80 + [2] * 20)
        + _split_with_grades("val", [0] * 70 + [2] * 30)
    )
    deviations = stratification_deviations(m)
    assert "eyepacs/val/2: 0.300 vs 0.250" in deviations
    assert len(deviations) == 4  # grades 0 and 2, in both splits


def test_grade_missing_from_a_split_counts_as_zero() -> None:
    m = _manifest(
        _split_with_grades("train", [0] * 50 + [4] * 50) + _split_with_grades("val", [0] * 100)
    )
    assert "eyepacs/val/4: 0.000 vs 0.250" in stratification_deviations(m)
