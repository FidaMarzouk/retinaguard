from pathlib import Path

import dagster as dg
import numpy as np
import pandas as pd
import pytest

from retinaguard.data.preprocess import PREPROCESS_VERSION
from retinaguard.data.sources import aptos, eyepacs
from retinaguard.pipeline.assets import DataPaths, asset_checks, assets

pytestmark = pytest.mark.integration


def _with_images(labels: pd.DataFrame, offset: int) -> pd.DataFrame:
    n = len(labels)
    return labels.assign(
        width=pd.array([800] * n, dtype="Int64"),
        height=pd.array([600] * n, dtype="Int64"),
        sha256=[f"{offset + i:064x}" for i in range(n)],
        phash=[f"{offset + i:016x}" for i in range(n)],
        processed_path=[f"shard_000.zip::{i}.jpg" for i in range(n)],
        preprocess_version=PREPROCESS_VERSION,
    )


def _candidates(pairs: list[tuple[str, str, float]]) -> pd.DataFrame:
    a, b, similarity = zip(*pairs, strict=True)
    return pd.DataFrame(
        {"image_id_a": a, "image_id_b": b, "phash_distance": 2, "similarity": similarity}
    )


def _interim(root: Path, aptos_pairs: list[tuple[str, str, float]], n_patients: int = 3000) -> None:
    rng = np.random.default_rng(0)
    names = [f"{p}_{eye}" for p in range(n_patients) for eye in ("left", "right")]
    grades = rng.choice(5, size=len(names), p=[0.73, 0.07, 0.15, 0.03, 0.02]).astype(str)
    eyepacs_manifest = _with_images(
        eyepacs.rows_from_labels(pd.DataFrame({"image": names, "level": grades})), 0
    )
    eyepacs_candidates = _candidates(
        [
            ("eyepacs:0_left", "eyepacs:1_right", 0.9),
            ("eyepacs:2_left", "eyepacs:3_left", 0.6),
            ("eyepacs:4_left", "eyepacs:5_left", 0.1),  # a look-alike, not a duplicate
        ]
    )
    codes = [f"{i:012x}" for i in range(200)]
    aptos_grades = [str(i % 5) for i in range(200)]
    aptos_manifest = _with_images(
        aptos.rows_from_labels(pd.DataFrame({"id_code": codes, "diagnosis": aptos_grades})),
        len(names),
    )
    for source, manifest, candidates_name, candidates in (
        ("eyepacs", eyepacs_manifest, "duplicate_candidates", eyepacs_candidates),
        ("aptos", aptos_manifest, "aptos_duplicate_candidates", _candidates(aptos_pairs)),
    ):
        interim = root / "interim" / source
        interim.mkdir(parents=True)
        manifest.to_parquet(interim / f"manifest_{source}.parquet", index=False)
        candidates.to_parquet(interim / f"{candidates_name}.parquet", index=False)


def _materialize(root: Path) -> dg.ExecuteInProcessResult:
    return dg.materialize([*assets, *asset_checks], resources={"data": DataPaths(root=str(root))})


def test_pipeline_writes_splits_and_checks_pass(tmp_path: Path) -> None:
    _interim(
        tmp_path,
        [
            ("aptos:000000000000", "aptos:000000000001", 1.0),  # grades 0 and 1
            ("aptos:000000000005", "aptos:00000000000a", 0.8),  # both grade 0
            ("aptos:000000000002", "eyepacs:7_left", 0.4),  # below the threshold
        ],
    )
    result = _materialize(tmp_path)

    assert result.success
    checks = result.get_asset_check_evaluations()
    assert len(checks) == len(asset_checks)
    assert all(check.passed for check in checks)

    manifest = pd.read_parquet(tmp_path / "manifests" / "manifest.parquet")
    eyepacs_splits = pd.read_parquet(tmp_path / "splits" / "eyepacs.parquet")
    aptos_splits = pd.read_parquet(tmp_path / "splits" / "aptos.parquet")
    assert len(eyepacs_splits) == 6000
    assert len(aptos_splits) == 200
    assert len(manifest) == 6200

    split_of = dict(zip(manifest["image_id"], manifest["split"], strict=True))
    assert split_of["eyepacs:0_left"] == split_of["eyepacs:1_right"]  # duplicate pair joined
    assert split_of["eyepacs:2_left"] == split_of["eyepacs:3_left"]
    groups = dict(zip(eyepacs_splits["image_id"], eyepacs_splits["split_group"], strict=True))
    assert groups["eyepacs:4_left"] != groups["eyepacs:5_left"]  # below the threshold

    assert (manifest.loc[manifest["source"] == "aptos", "split"] == "test_external").all()
    aptos_groups = aptos_splits.set_index("image_id")["split_group"]
    assert aptos_groups["aptos:000000000001"] == "aptos:000000000000"
    assert aptos_splits["split_group"].nunique() == 198

    metadata = result.asset_materializations_for_node("aptos_splits")[0].metadata
    assert metadata["duplicate_groups"].value == 2
    assert metadata["groups_with_different_grades"].value == 1


def test_external_duplicate_of_development_image_fails_check(tmp_path: Path) -> None:
    _interim(tmp_path, [("aptos:000000000003", "eyepacs:7_left", 0.95)])
    result = _materialize(tmp_path)

    failed = [c.check_name for c in result.get_asset_check_evaluations() if not c.passed]
    assert failed == ["no_duplicate_across_splits"]
