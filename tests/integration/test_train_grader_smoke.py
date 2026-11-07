import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from retinaguard.config import load_config
from retinaguard.train.evaluate import evaluate
from retinaguard.train.predict import PROBABILITY_COLUMNS
from retinaguard.train.predict import main as predict_main
from retinaguard.train.train_grader import train

pytestmark = pytest.mark.smoke


def test_train_resume_predict_evaluate(
    tmp_path: Path, fixture_shard: tuple[Path, pd.DataFrame]
) -> None:
    shard_dir, manifest = fixture_shard
    manifest.to_parquet(tmp_path / "manifest.parquet")
    overrides = [
        f"data.manifest={(tmp_path / 'manifest.parquet').as_posix()}",
        f"data.shard_dirs=[{shard_dir.as_posix()}]",
        f"output_dir={(tmp_path / 'runs').as_posix()}",
        "model.pretrained=false",
        "data.image_size=128",
        "train.batch_size=16",
        "train.num_workers=0",
        "train.device=cpu",
        "train.imbalance=sampler",
    ]
    run_dir = train(load_config("train/grader", [*overrides, "train.epochs=1"]))

    assert (run_dir / "best.pt").exists()
    assert torch.load(run_dir / "last.pt", weights_only=False)["epoch"] == 0
    # a second call with more epochs continues after epoch 0 instead of starting over
    train(load_config("train/grader", [*overrides, "train.epochs=2"]))
    lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    steps = sorted({json.loads(line)["step"] for line in lines})
    assert steps == [0, 1]
    metric_rows = [json.loads(line) for line in lines]
    assert sum(r["key"] == "val_referable_auroc" for r in metric_rows) == 2
    assert torch.load(run_dir / "best.pt", weights_only=False)["config"]["model"]["name"] == (
        "convnext_tiny"
    )

    # predict with the selected weights, then evaluate as if val were the internal test split
    test_manifest = manifest.assign(
        split=manifest["split"].replace({"val": "test_internal"}),
        split_group=manifest["image_id"],
        source="eyepacs",
    )
    test_manifest.to_parquet(tmp_path / "test_manifest.parquet")
    predict_main(
        [
            f"--checkpoint={run_dir / 'best.pt'}",
            f"--manifest={tmp_path / 'test_manifest.parquet'}",
            f"--shard-dirs={shard_dir}",
            f"--out-dir={tmp_path / 'predictions'}",
            "--num-workers=0",
            "--device=cpu",
        ]
    )
    predictions = pd.read_parquet(tmp_path / "predictions" / "predictions.parquet")
    embeddings = np.load(tmp_path / "predictions" / "embeddings.npy")
    assert len(predictions) == len(embeddings) == 16
    assert np.allclose(predictions[PROBABILITY_COLUMNS].sum(axis=1), 1, atol=1e-4)
    table = evaluate(test_manifest, predictions, n_resamples=20)
    assert table.loc["eyepacs_test", "images"] == 16
    assert 0 <= table.loc["eyepacs_test", "auroc"] <= 1
