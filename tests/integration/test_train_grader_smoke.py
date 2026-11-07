import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from retinaguard.config import load_config
from retinaguard.train.train_grader import train

pytestmark = pytest.mark.smoke


def test_one_epoch_on_cpu_then_resume(
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
