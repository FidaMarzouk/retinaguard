"""Data assets: per-source processed-image manifests -> confirmed duplicates -> frozen splits ->
one combined manifest, on which the data invariants are checked.

Images are preprocessed and near-duplicate candidates scored in Kaggle notebooks; their small
outputs are placed under ``data/interim/{source}/``:

- EyePACS: ``manifest_eyepacs.parquet``, ``duplicate_candidates.parquet``
- APTOS: ``manifest_aptos.parquet``, ``aptos_duplicate_candidates.parquet`` (pairs within APTOS
  and between APTOS and EyePACS)

The split files and the combined manifest are written under ``data/`` and versioned with DVC.
"""

from pathlib import Path

import dagster as dg
import pandas as pd

from retinaguard.data import invariants
from retinaguard.data.duplicates import DUPLICATE_MIN_SIMILARITY, confirmed_pairs
from retinaguard.data.schemas import ImageManifestSchema, SplitManifestSchema
from retinaguard.data.splits import (
    SPLIT_SEED,
    assign_splits,
    conflicting_groups,
    external_splits,
)


class DataPaths(dg.ConfigurableResource):  # type: ignore[type-arg]
    root: str = "data"

    def interim(self, source: str) -> Path:
        return Path(self.root) / "interim" / source

    def splits_file(self, source: str) -> Path:
        return Path(self.root) / "splits" / f"{source}.parquet"

    @property
    def manifest_file(self) -> Path:
        return Path(self.root) / "manifests" / "manifest.parquet"


def _write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _confirmed(context: dg.AssetExecutionContext, candidates_file: Path) -> pd.DataFrame:
    candidates = pd.read_parquet(candidates_file)
    pairs = pd.DataFrame(confirmed_pairs(candidates), columns=["image_id_a", "image_id_b"])
    context.add_output_metadata(
        {
            "candidate_pairs": len(candidates),
            "confirmed_pairs": len(pairs),
            "min_similarity": DUPLICATE_MIN_SIMILARITY,
        }
    )
    return pairs


def _pairs(frame: pd.DataFrame) -> list[tuple[str, str]]:
    return list(zip(frame["image_id_a"], frame["image_id_b"], strict=True))


@dg.asset(group_name="eyepacs")
def eyepacs_manifest(data: DataPaths) -> pd.DataFrame:
    """Labels plus image columns from the preprocessing notebook, schema-validated."""
    manifest = pd.read_parquet(data.interim("eyepacs") / "manifest_eyepacs.parquet")
    return ImageManifestSchema.validate(manifest)


@dg.asset(group_name="eyepacs")
def eyepacs_duplicate_pairs(
    context: dg.AssetExecutionContext, data: DataPaths, eyepacs_manifest: pd.DataFrame
) -> pd.DataFrame:
    """Candidate pairs from the duplicates notebook whose structure similarity passes."""
    return _confirmed(context, data.interim("eyepacs") / "duplicate_candidates.parquet")


@dg.asset(group_name="eyepacs")
def eyepacs_splits(
    context: dg.AssetExecutionContext,
    data: DataPaths,
    eyepacs_manifest: pd.DataFrame,
    eyepacs_duplicate_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Manifest with a patient-level, grade-stratified split; writes the frozen split file."""
    splits = assign_splits(eyepacs_manifest, _pairs(eyepacs_duplicate_pairs))
    manifest = SplitManifestSchema.validate(
        eyepacs_manifest.assign(split=splits["split"].to_numpy())
    )
    _write(data.splits_file("eyepacs"), splits)
    context.add_output_metadata(
        {
            "seed": SPLIT_SEED,
            "images_per_split": manifest["split"].value_counts().sort_index().to_dict(),
            "patients_per_split": manifest.groupby("split")["patient_id"].nunique().to_dict(),
            "largest_group": int(splits["split_group"].value_counts().max()),
        }
    )
    return manifest


@dg.asset(group_name="aptos")
def aptos_manifest(data: DataPaths) -> pd.DataFrame:
    """Labels plus image columns from the APTOS preprocessing notebook, schema-validated."""
    manifest = pd.read_parquet(data.interim("aptos") / "manifest_aptos.parquet")
    return ImageManifestSchema.validate(manifest)


@dg.asset(group_name="aptos")
def aptos_duplicate_pairs(
    context: dg.AssetExecutionContext, data: DataPaths, aptos_manifest: pd.DataFrame
) -> pd.DataFrame:
    """Confirmed duplicates within APTOS and between APTOS and EyePACS."""
    return _confirmed(context, data.interim("aptos") / "aptos_duplicate_candidates.parquet")


@dg.asset(group_name="aptos")
def aptos_splits(
    context: dg.AssetExecutionContext,
    data: DataPaths,
    aptos_manifest: pd.DataFrame,
    aptos_duplicate_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Manifest with every image in ``test_external``; duplicates of one eye share a group."""
    splits = external_splits(aptos_manifest, _pairs(aptos_duplicate_pairs))
    manifest = SplitManifestSchema.validate(aptos_manifest.assign(split=splits["split"].to_numpy()))
    _write(data.splits_file("aptos"), splits)
    group_sizes = splits["split_group"].value_counts()
    groups = splits["split_group"]
    context.add_output_metadata(
        {
            "images": len(manifest),
            "duplicate_groups": int((group_sizes > 1).sum()),
            "images_in_duplicate_groups": int(group_sizes[group_sizes > 1].sum()),
            "groups_with_different_grades": len(conflicting_groups(manifest, groups, "dr_grade")),
            "groups_with_different_referable": len(
                conflicting_groups(manifest, groups, "referable")
            ),
        }
    )
    return manifest


@dg.asset(group_name="combined")
def manifest(
    context: dg.AssetExecutionContext,
    data: DataPaths,
    eyepacs_splits: pd.DataFrame,
    aptos_splits: pd.DataFrame,
) -> pd.DataFrame:
    """All sources with their splits in one validated manifest, the input to training."""
    combined = SplitManifestSchema.validate(
        pd.concat([eyepacs_splits, aptos_splits], ignore_index=True)
    )
    _write(data.manifest_file, combined)
    context.add_output_metadata(
        {"images_per_source_split": combined.groupby(["source", "split"]).size().to_string()}
    )
    return combined


def _check(violations: list[str]) -> dg.AssetCheckResult:
    return dg.AssetCheckResult(
        passed=not violations,
        metadata={"violations": len(violations), "examples": violations[:10]},
    )


@dg.asset_check(asset=manifest)
def unique_image_ids(manifest: pd.DataFrame) -> dg.AssetCheckResult:
    return _check(invariants.duplicate_image_ids(manifest))


@dg.asset_check(asset=manifest)
def no_patient_in_two_splits(manifest: pd.DataFrame) -> dg.AssetCheckResult:
    return _check(invariants.patients_in_several_splits(manifest))


@dg.asset_check(
    asset=manifest,
    additional_ins={
        "eyepacs_pairs": dg.AssetIn("eyepacs_duplicate_pairs"),
        "aptos_pairs": dg.AssetIn("aptos_duplicate_pairs"),
    },
)
def no_duplicate_across_splits(
    manifest: pd.DataFrame, eyepacs_pairs: pd.DataFrame, aptos_pairs: pd.DataFrame
) -> dg.AssetCheckResult:
    pairs = _pairs(eyepacs_pairs) + _pairs(aptos_pairs)
    return _check(invariants.duplicates_across_splits(manifest, pairs))


@dg.asset_check(asset=manifest)
def no_external_in_training(manifest: pd.DataFrame) -> dg.AssetCheckResult:
    return _check(invariants.external_images_in_training(manifest))


@dg.asset_check(asset=manifest)
def stratified_by_grade(manifest: pd.DataFrame) -> dg.AssetCheckResult:
    return _check(invariants.stratification_deviations(manifest))


assets = [
    eyepacs_manifest,
    eyepacs_duplicate_pairs,
    eyepacs_splits,
    aptos_manifest,
    aptos_duplicate_pairs,
    aptos_splits,
    manifest,
]
asset_checks = [
    unique_image_ids,
    no_patient_in_two_splits,
    no_duplicate_across_splits,
    no_external_in_training,
    stratified_by_grade,
]
