"""Pandera schemas for the image manifest.

Columns are added stage by stage: source adapters produce ``LabelManifestSchema`` rows from
label files alone; image-derived columns (size, hashes, processed path, split) are added by
later pipeline steps.
"""

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from retinaguard.data.labels import ICDR_GRADES, REFERABLE_MIN_GRADE

SOURCES = ("eyepacs", "eyeq", "aptos", "messidor2", "idrid", "ddr")
SITE_COUNTRIES = ("US", "IN", "FR", "CN")
EYES = ("left", "right")
QUALITY_LABELS = ("good", "usable", "reject")
SPLITS = ("train", "val", "calib", "test_internal", "test_external")

# Grading sources used only for external testing (see ``invariants.external_images_in_training``).
EXTERNAL_SOURCES = ("aptos", "messidor2", "idrid", "ddr")
TRAINING_SPLITS = ("train", "val", "calib")


class LabelManifestSchema(pa.DataFrameModel):
    image_id: Series[str] = pa.Field(unique=True, str_matches=r"^[a-z0-9]+:.+$")
    source: Series[str] = pa.Field(isin=SOURCES)
    site_country: Series[str] = pa.Field(isin=SITE_COUNTRIES)
    patient_id: Series[str] = pa.Field(nullable=True)
    eye: Series[str] = pa.Field(nullable=True, isin=EYES)
    raw_path: Series[str] = pa.Field(str_length={"min_value": 1})
    dr_grade: Series[pd.Int64Dtype] = pa.Field(nullable=True, isin=list(ICDR_GRADES))
    referable: Series[pd.BooleanDtype] = pa.Field(nullable=True)
    quality_label: Series[str] = pa.Field(nullable=True, isin=QUALITY_LABELS)
    has_lesion_mask: Series[bool]
    label_raw: Series[str]
    label_map_version: Series[str]

    @pa.dataframe_check
    @classmethod
    def referable_matches_grade(cls, df: pd.DataFrame) -> "pd.Series[bool]":
        graded = df["dr_grade"].notna()
        expected = df["dr_grade"] >= REFERABLE_MIN_GRADE
        return ~graded | (df["referable"] == expected).fillna(False)

    @pa.dataframe_check
    @classmethod
    def ids_prefixed_by_source(cls, df: pd.DataFrame) -> "pd.Series[bool]":
        id_ok = df["image_id"].str.split(":", n=1).str[0] == df["source"]
        patient_ok = df["patient_id"].isna() | (
            df["patient_id"].str.split(":", n=1).str[0] == df["source"]
        )
        return id_ok & patient_ok

    class Config:
        strict = True
        ordered = True


class ImageManifestSchema(LabelManifestSchema):
    """Label rows plus the image-derived columns filled by preprocessing.

    Images that failed preprocessing stay in the manifest with null ``phash``/``processed_path``,
    so failures remain visible downstream instead of silently shrinking the dataset.
    """

    width: Series[pd.Int64Dtype] = pa.Field(nullable=True, gt=0)
    height: Series[pd.Int64Dtype] = pa.Field(nullable=True, gt=0)
    sha256: Series[str] = pa.Field(str_matches=r"^[0-9a-f]{64}$")
    phash: Series[str] = pa.Field(nullable=True, str_matches=r"^[0-9a-f]{16}$")
    processed_path: Series[str] = pa.Field(nullable=True)
    preprocess_version: Series[str]

    @pa.dataframe_check
    @classmethod
    def processed_iff_hashed(cls, df: pd.DataFrame) -> "pd.Series[bool]":
        return df["processed_path"].isna() == df["phash"].isna()


class SplitManifestSchema(ImageManifestSchema):
    """Image rows with their assigned split."""

    split: Series[str] = pa.Field(isin=SPLITS)
