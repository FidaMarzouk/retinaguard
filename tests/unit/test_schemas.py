import pandas as pd
import pandera.errors
import pytest

from retinaguard.data.schemas import LabelManifestSchema
from retinaguard.data.sources import eyepacs


@pytest.fixture
def rows() -> pd.DataFrame:
    labels = pd.DataFrame({"image": ["1_left", "1_right"], "level": ["0", "3"]}, dtype=str)
    return eyepacs.rows_from_labels(labels)


def test_valid_rows_pass(rows: pd.DataFrame) -> None:
    LabelManifestSchema.validate(rows)


def test_referable_must_match_grade(rows: pd.DataFrame) -> None:
    rows.loc[1, "referable"] = False
    with pytest.raises(pandera.errors.SchemaError, match="referable_matches_grade"):
        LabelManifestSchema.validate(rows)


def test_ungraded_rows_allowed(rows: pd.DataFrame) -> None:
    rows.loc[0, ["dr_grade", "referable"]] = pd.NA
    LabelManifestSchema.validate(rows)


def test_ids_must_carry_source_prefix(rows: pd.DataFrame) -> None:
    rows.loc[0, "patient_id"] = "aptos:1"
    with pytest.raises(pandera.errors.SchemaError, match="ids_prefixed_by_source"):
        LabelManifestSchema.validate(rows)


@pytest.mark.parametrize(
    ("column", "value"),
    [("source", "kaggle"), ("site_country", "UK"), ("eye", "both"), ("quality_label", "bad")],
)
def test_category_values_checked(rows: pd.DataFrame, column: str, value: str) -> None:
    rows.loc[0, column] = value
    with pytest.raises(pandera.errors.SchemaError):
        LabelManifestSchema.validate(rows)


def test_unknown_columns_rejected(rows: pd.DataFrame) -> None:
    # strict and ordered both flag the column, so pandera reports them together.
    with pytest.raises((pandera.errors.SchemaError, pandera.errors.SchemaErrors)):
        LabelManifestSchema.validate(rows.assign(extra=1))
