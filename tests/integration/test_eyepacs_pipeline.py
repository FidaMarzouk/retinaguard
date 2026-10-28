from pathlib import Path

import dagster as dg
import numpy as np
import pandas as pd
import pytest

from retinaguard.data.preprocess import PREPROCESS_VERSION
from retinaguard.data.sources import eyepacs
from retinaguard.pipeline.assets import DataPaths, asset_checks, assets

pytestmark = pytest.mark.integration


def _interim(root: Path, n_patients: int = 3000) -> None:
    rng = np.random.default_rng(0)
    names = [f"{p}_{eye}" for p in range(n_patients) for eye in ("left", "right")]
    grades = rng.choice(5, size=len(names), p=[0.73, 0.07, 0.15, 0.03, 0.02]).astype(str)
    manifest = eyepacs.rows_from_labels(pd.DataFrame({"image": names, "level": grades}))
    manifest = manifest.assign(
        width=pd.array([800] * len(names), dtype="Int64"),
        height=pd.array([600] * len(names), dtype="Int64"),
        sha256=[f"{i:064x}" for i in range(len(names))],
        phash=[f"{i:016x}" for i in range(len(names))],
        processed_path=[f"eyepacs_000.zip::{n}.jpg" for n in names],
        preprocess_version=PREPROCESS_VERSION,
    )
    candidates = pd.DataFrame(
        {
            "image_id_a": ["eyepacs:0_left", "eyepacs:2_left", "eyepacs:4_left"],
            "image_id_b": ["eyepacs:1_left", "eyepacs:3_right", "eyepacs:5_left"],
            "phash_distance": [2, 4, 0],
            "same_patient": False,
            "similarity": [0.9, 0.6, 0.1],  # the last one is a look-alike, not a duplicate
        }
    )
    interim = root / "interim" / "eyepacs"
    interim.mkdir(parents=True)
    manifest.to_parquet(interim / "manifest_eyepacs.parquet", index=False)
    candidates.to_parquet(interim / "duplicate_candidates.parquet", index=False)


def test_pipeline_writes_splits_and_checks_pass(tmp_path: Path) -> None:
    _interim(tmp_path)
    result = dg.materialize(
        [*assets, *asset_checks], resources={"data": DataPaths(root=str(tmp_path))}
    )

    assert result.success
    checks = result.get_asset_check_evaluations()
    assert len(checks) == len(asset_checks)
    assert all(check.passed for check in checks)

    manifest = pd.read_parquet(tmp_path / "manifests" / "manifest.parquet")
    splits = pd.read_parquet(tmp_path / "splits" / "eyepacs.parquet")
    assert len(manifest) == len(splits) == 6000
    split_of = dict(zip(manifest["image_id"], manifest["split"], strict=True))
    assert split_of["eyepacs:0_left"] == split_of["eyepacs:1_right"]  # duplicate pair joined
    assert split_of["eyepacs:2_left"] == split_of["eyepacs:3_left"]
    groups = dict(zip(splits["image_id"], splits["split_group"], strict=True))
    assert groups["eyepacs:4_left"] != groups["eyepacs:5_left"]  # below the threshold
