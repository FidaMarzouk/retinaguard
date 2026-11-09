import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from retinaguard.config import load_config
from retinaguard.models.quality import QUALITY_CLASSES, QUALITY_COLUMNS
from retinaguard.train.evaluate_quality import evaluate_quality
from retinaguard.train.train_quality import predict_run, train

pytestmark = pytest.mark.smoke


def test_train_resume_predict_evaluate_quality(
    tmp_path: Path, fixture_shard: tuple[Path, pd.DataFrame]
) -> None:
    shard_dir, manifest = fixture_shard
    labels = [QUALITY_CLASSES[i % 3] for i in range(len(manifest))]
    manifest = manifest.assign(
        source="eyepacs", quality_label=labels, split_group=manifest["image_id"]
    )
    manifest.to_parquet(tmp_path / "manifest.parquet")
    overrides = [
        f"data.manifest={(tmp_path / 'manifest.parquet').as_posix()}",
        f"data.shard_dirs=[{shard_dir.as_posix()}]",
        f"output_dir={(tmp_path / 'runs').as_posix()}",
        "model.timm_name=test_efficientnet",
        "model.pretrained=false",
        "data.image_size=128",
        "train.batch_size=16",
        "train.num_workers=0",
        "train.device=cpu",
    ]
    run_dir = train(load_config("train/quality", [*overrides, "train.epochs=1"]))
    assert torch.load(run_dir / "last.pt", weights_only=False)["epoch"] == 0
    # a second call with more epochs continues after epoch 0 instead of starting over
    cfg = load_config("train/quality", [*overrides, "train.epochs=2"])
    train(cfg)
    lines = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert sorted({line["step"] for line in lines}) == [0, 1]
    assert sum(line["key"] == "val_macro_f1" for line in lines) == 2

    predictions = predict_run(cfg, run_dir)
    assert len(predictions) == 16  # val only: train images are not predicted
    assert np.allclose(predictions[QUALITY_COLUMNS].sum(axis=1), 1, atol=1e-4)
    table = evaluate_quality(manifest, predictions, n_resamples=20)
    assert table.loc["eyepacs_val", "images"] == 16
    assert 0 <= table.loc["eyepacs_val", "macro_f1"] <= 1
