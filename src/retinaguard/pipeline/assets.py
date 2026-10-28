"""EyePACS data assets: processed-image manifest -> confirmed duplicates -> frozen splits.

Images are preprocessed and near-duplicate candidates scored in Kaggle notebooks; their small
outputs (``manifest_eyepacs.parquet``, ``duplicate_candidates.parquet``) are placed under
``data/interim/eyepacs/``. The split file and the split manifest are written under ``data/`` and
versioned with DVC.
"""

from pathlib import Path

import dagster as dg
import pandas as pd

from retinaguard.data import invariants
from retinaguard.data.duplicates import DUPLICATE_MIN_SIMILARITY, confirmed_pairs
from retinaguard.data.schemas import ImageManifestSchema, SplitManifestSchema
from retinaguard.data.splits import SPLIT_SEED, assign_splits


class DataPaths(dg.ConfigurableResource):  # type: ignore[type-arg]
    root: str = "data"

    @property
    def interim(self) -> Path:
        return Path(self.root) / "interim" / "eyepacs"

    @property
    def splits_file(self) -> Path:
        return Path(self.root) / "splits" / "eyepacs.parquet"

    @property
    def manifest_file(self) -> Path:
        return Path(self.root) / "manifests" / "manifest.parquet"


@dg.asset(group_name="eyepacs")
def eyepacs_manifest(data: DataPaths) -> pd.DataFrame:
    """Labels plus image columns from the preprocessing notebook, schema-validated."""
    manifest = pd.read_parquet(data.interim / "manifest_eyepacs.parquet")
    return ImageManifestSchema.validate(manifest)


@dg.asset(group_name="eyepacs")
def eyepacs_duplicate_pairs(
    context: dg.AssetExecutionContext, data: DataPaths, eyepacs_manifest: pd.DataFrame
) -> pd.DataFrame:
    """Candidate pairs from the duplicates notebook whose structure similarity passes."""
    candidates = pd.read_parquet(data.interim / "duplicate_candidates.parquet")
    pairs = pd.DataFrame(confirmed_pairs(candidates), columns=["image_id_a", "image_id_b"])
    context.add_output_metadata(
        {
            "candidate_pairs": len(candidates),
            "confirmed_pairs": len(pairs),
            "min_similarity": DUPLICATE_MIN_SIMILARITY,
        }
    )
    return pairs


@dg.asset(group_name="eyepacs")
def eyepacs_splits(
    context: dg.AssetExecutionContext,
    data: DataPaths,
    eyepacs_manifest: pd.DataFrame,
    eyepacs_duplicate_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Manifest with a patient-level, grade-stratified split; writes the frozen split files."""
    pairs = _pairs(eyepacs_duplicate_pairs)
    splits = assign_splits(eyepacs_manifest, pairs)
    manifest = SplitManifestSchema.validate(
        eyepacs_manifest.assign(split=splits["split"].to_numpy())
    )
    for path, frame in ((data.splits_file, splits), (data.manifest_file, manifest)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    context.add_output_metadata(
        {
            "seed": SPLIT_SEED,
            "images_per_split": manifest["split"].value_counts().sort_index().to_dict(),
            "patients_per_split": manifest.groupby("split")["patient_id"].nunique().to_dict(),
            "largest_group": int(splits["split_group"].value_counts().max()),
        }
    )
    return manifest


def _pairs(frame: pd.DataFrame) -> list[tuple[str, str]]:
    return list(zip(frame["image_id_a"], frame["image_id_b"], strict=True))


def _check(violations: list[str]) -> dg.AssetCheckResult:
    return dg.AssetCheckResult(
        passed=not violations,
        metadata={"violations": len(violations), "examples": violations[:10]},
    )


@dg.asset_check(asset=eyepacs_manifest)
def unique_image_ids(eyepacs_manifest: pd.DataFrame) -> dg.AssetCheckResult:
    return _check(invariants.duplicate_image_ids(eyepacs_manifest))


@dg.asset_check(asset=eyepacs_splits)
def no_patient_in_two_splits(eyepacs_splits: pd.DataFrame) -> dg.AssetCheckResult:
    return _check(invariants.patients_in_several_splits(eyepacs_splits))


@dg.asset_check(
    asset=eyepacs_splits, additional_ins={"pairs": dg.AssetIn("eyepacs_duplicate_pairs")}
)
def no_duplicate_across_splits(
    eyepacs_splits: pd.DataFrame, pairs: pd.DataFrame
) -> dg.AssetCheckResult:
    return _check(invariants.duplicates_across_splits(eyepacs_splits, _pairs(pairs)))


@dg.asset_check(asset=eyepacs_splits)
def no_external_in_training(eyepacs_splits: pd.DataFrame) -> dg.AssetCheckResult:
    return _check(invariants.external_images_in_training(eyepacs_splits))


@dg.asset_check(asset=eyepacs_splits)
def stratified_by_grade(eyepacs_splits: pd.DataFrame) -> dg.AssetCheckResult:
    return _check(invariants.stratification_deviations(eyepacs_splits))


assets = [eyepacs_manifest, eyepacs_duplicate_pairs, eyepacs_splits]
asset_checks = [
    unique_image_ids,
    no_patient_in_two_splits,
    no_duplicate_across_splits,
    no_external_in_training,
    stratified_by_grade,
]
