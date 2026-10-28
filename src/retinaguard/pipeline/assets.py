"""Data assets: per-source processed-image manifests -> confirmed duplicates -> frozen splits ->
one combined manifest, on which the data invariants are checked.

Images are preprocessed and near-duplicate candidates scored in Kaggle notebooks; their small
outputs are placed under ``data/interim/{source}/``:

- EyePACS: ``manifest_eyepacs.parquet``, ``duplicate_candidates.parquet``
- each external test source (``EXTERNAL_TEST_SOURCES``): ``manifest_{source}.parquet`` and
  ``{source}_duplicate_candidates.parquet``, with pairs within the source and against every source
  processed before it

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


# Processing order: each source's duplicate candidates cover the sources before it.
EXTERNAL_TEST_SOURCES = ("aptos", "idrid")


def _external_source(source: str) -> list[dg.AssetsDefinition]:
    """Manifest, confirmed duplicates and splits of an external test source."""

    @dg.asset(name=f"{source}_manifest", group_name=source)
    def source_manifest(data: DataPaths) -> pd.DataFrame:
        """Labels plus image columns from the preprocessing notebook, schema-validated."""
        manifest = pd.read_parquet(data.interim(source) / f"manifest_{source}.parquet")
        return ImageManifestSchema.validate(manifest)

    @dg.asset(name=f"{source}_duplicate_pairs", group_name=source, deps=[source_manifest])
    def duplicate_pairs(context: dg.AssetExecutionContext, data: DataPaths) -> pd.DataFrame:
        """Confirmed duplicates within the source and against earlier sources."""
        return _confirmed(context, data.interim(source) / f"{source}_duplicate_candidates.parquet")

    @dg.asset(
        name=f"{source}_splits",
        group_name=source,
        ins={
            "labels": dg.AssetIn(f"{source}_manifest"),
            "pairs": dg.AssetIn(f"{source}_duplicate_pairs"),
        },
    )
    def splits(
        context: dg.AssetExecutionContext,
        data: DataPaths,
        labels: pd.DataFrame,
        pairs: pd.DataFrame,
    ) -> pd.DataFrame:
        """Manifest with every image in ``test_external``; duplicates of one eye share a group."""
        frame = external_splits(labels, _pairs(pairs))
        manifest = SplitManifestSchema.validate(labels.assign(split=frame["split"].to_numpy()))
        _write(data.splits_file(source), frame)
        groups = frame["split_group"]
        group_sizes = groups.value_counts()
        context.add_output_metadata(
            {
                "images": len(manifest),
                "duplicate_groups": int((group_sizes > 1).sum()),
                "images_in_duplicate_groups": int(group_sizes[group_sizes > 1].sum()),
                "groups_with_different_grades": len(
                    conflicting_groups(manifest, groups, "dr_grade")
                ),
                "groups_with_different_referable": len(
                    conflicting_groups(manifest, groups, "referable")
                ),
            }
        )
        return manifest

    return [source_manifest, duplicate_pairs, splits]


ALL_SOURCES = ("eyepacs", *EXTERNAL_TEST_SOURCES)


@dg.asset(
    group_name="combined",
    ins={source: dg.AssetIn(f"{source}_splits") for source in ALL_SOURCES},
)
def manifest(
    context: dg.AssetExecutionContext, data: DataPaths, **splits: pd.DataFrame
) -> pd.DataFrame:
    """All sources with their splits in one validated manifest, the input to training."""
    combined = SplitManifestSchema.validate(pd.concat(splits.values(), ignore_index=True))
    _write(data.manifest_file, combined)
    context.add_output_metadata(
        {"images_per_source_split": combined.groupby(["source", "split"]).size().to_string()}
    )
    return combined


@dg.asset(
    group_name="combined",
    ins={source: dg.AssetIn(f"{source}_duplicate_pairs") for source in ALL_SOURCES},
)
def duplicate_pairs(**pairs: pd.DataFrame) -> pd.DataFrame:
    """Confirmed duplicate pairs of all sources, within and across datasets."""
    return pd.concat(pairs.values(), ignore_index=True)


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


@dg.asset_check(asset=manifest, additional_ins={"pairs": dg.AssetIn("duplicate_pairs")})
def no_duplicate_across_splits(manifest: pd.DataFrame, pairs: pd.DataFrame) -> dg.AssetCheckResult:
    return _check(invariants.duplicates_across_splits(manifest, _pairs(pairs)))


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
    *(asset for source in EXTERNAL_TEST_SOURCES for asset in _external_source(source)),
    manifest,
    duplicate_pairs,
]
asset_checks = [
    unique_image_ids,
    no_patient_in_two_splits,
    no_duplicate_across_splits,
    no_external_in_training,
    stratified_by_grade,
]
