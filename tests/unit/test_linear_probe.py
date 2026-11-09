from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from retinaguard.config import load_config
from retinaguard.models.grader import Grader
from retinaguard.train.extract_embeddings import embedding_rows, extract
from retinaguard.train.linear_probe import fit_probe, main, probe_predictions
from retinaguard.train.predict import PROBABILITY_COLUMNS
from retinaguard.train.train_grader import flatten


def _embedding_dir(root: Path, n: int = 400, dim: int = 16) -> tuple[Path, pd.DataFrame]:
    """Embeddings whose first coordinate tracks the grade, for a manifest of n images."""
    rng = np.random.default_rng(0)
    grades = rng.integers(0, 5, n)
    split = np.where(
        np.arange(n) < 250, "train", np.where(np.arange(n) < 320, "val", "test_internal")
    )
    manifest = pd.DataFrame(
        {
            "image_id": [f"eyepacs:{i:04d}" for i in range(n)],
            "split": split,
            "processed_path": [f"s.zip::{i}.jpg" for i in range(n)],
            "dr_grade": pd.array(grades, dtype="Int64"),
            "quality_label": None,
        }
    )
    features = rng.normal(size=(n, dim))
    features[:, 0] += 2.0 * grades
    folder = root / "embeddings"
    folder.mkdir()
    order = rng.permutation(n)  # stored order differs from manifest order
    np.save(folder / "embeddings.npy", features[order].astype(np.float16))
    manifest.iloc[order][["image_id"]].to_parquet(folder / "index.parquet", index=False)
    (folder / "meta.json").write_text('{"model": "synthetic", "image_size": 512}')
    return folder, manifest


def test_probe_predictions_format() -> None:
    rng = np.random.default_rng(1)
    grades = np.repeat([0, 2, 4], 20)  # grades 1 and 3 never seen in training
    features = rng.normal(size=(60, 4)) + grades[:, None]
    grade_model, referable_model = fit_probe(features, grades, c=1.0)
    out = probe_predictions(grade_model, referable_model, features, pd.Series(range(60)))
    assert list(out.columns) == ["image_id", *PROBABILITY_COLUMNS, "p_referable"]
    assert np.allclose(out[PROBABILITY_COLUMNS].sum(axis=1), 1)
    assert (out[["p_grade_1", "p_grade_3"]] == 0).all().all()


def test_linear_probe_main(tmp_path: Path) -> None:
    folder, manifest = _embedding_dir(tmp_path)
    manifest.to_parquet(tmp_path / "manifest.parquet")
    out_dir = tmp_path / "predictions" / "probe-synthetic"
    main(
        [
            f"--manifest={tmp_path / 'manifest.parquet'}",
            f"--embeddings={folder}",
            f"--out-dir={out_dir}",
            f"--runs-dir={tmp_path / 'runs'}",
        ]
    )
    predictions = pd.read_parquet(out_dir / "predictions.parquet")
    assert set(predictions["image_id"]) == set(
        manifest.loc[manifest["split"] != "train", "image_id"]
    )
    joined = predictions.merge(manifest, on="image_id")
    referable = joined["dr_grade"].astype(int) >= 2
    assert (
        joined.loc[referable, "p_referable"].mean()
        > joined.loc[~referable, "p_referable"].mean() + 0.3
    )
    assert np.load(out_dir / "embeddings.npy").shape == (len(predictions), 16)
    assert (tmp_path / "runs" / "probe-synthetic" / "metrics.jsonl").exists()


def test_extract_embeddings(fixture_shard: tuple[Path, pd.DataFrame]) -> None:
    shard_dir, manifest = fixture_shard
    rows = embedding_rows(manifest)
    backbone = Grader(
        "test_vit",
        pretrained=False,
        timm_kwargs={"dynamic_img_size": True, "dynamic_img_pad": True},
    ).backbone
    embeddings = extract(
        backbone,
        rows,
        [shard_dir],
        image_size=64,
        local_mean_norm=False,
        num_workers=0,
        device=torch.device("cpu"),
    )
    assert embeddings.shape == (len(manifest), backbone.num_features)
    assert embeddings.dtype == np.float16


@pytest.mark.parametrize("size", [512, 518])
def test_dinov2_config_accepts_512_inputs(size: int) -> None:
    cfg = load_config("train/grader", ["model=dinov2_vitb"])
    assert flatten(dict(cfg))["model.timm_kwargs.dynamic_img_pad"] is True
    model = Grader(
        cfg.model.timm_name,
        pretrained=False,
        timm_kwargs={**cfg.model.timm_kwargs, "depth": 1},  # one block keeps the test fast
    )
    output = model(torch.zeros(1, 3, size, size))
    assert output.embedding.shape == (1, 768)
