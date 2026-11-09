import json
from pathlib import Path

import numpy as np
import pytest
import torch

from retinaguard.config import load_config
from retinaguard.train.train_seg import predict_run, train

pytestmark = pytest.mark.smoke


def test_train_resume_predict_seg(tmp_path: Path, seg_shard: Path) -> None:
    overrides = [
        f"data.shard_dirs=[{seg_shard.as_posix()}]",
        f"output_dir={(tmp_path / 'runs').as_posix()}",
        "run_name=seg-idrid_only",
        "data.train_sites=[idrid]",
        "model.encoder=resnet18",
        "model.pretrained=false",
        "data.crop_size=64",
        "train.samples_per_epoch=8",
        "train.batch_size=4",
        "train.num_workers=0",
        "train.val_every=1",
        "train.device=cpu",
    ]
    run_dir = train(load_config("train/segmentation", [*overrides, "train.epochs=1"]))
    assert torch.load(run_dir / "last.pt", weights_only=False)["epoch"] == 0
    cfg = load_config("train/segmentation", [*overrides, "train.epochs=2"])
    train(cfg)
    lines = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert sorted({line["step"] for line in lines}) == [0, 1]
    assert sum(line["key"] == "val_score" for line in lines) == 2

    rows = predict_run(cfg, run_dir)
    assert set(rows["split"]) == {"val", "test"}  # training images are not segmented
    with np.load(run_dir / "stats.npz", allow_pickle=False) as stats:
        assert stats["image_id"].dtype.kind == "U"
        assert stats["tp"].shape == (len(rows), 4)
        assert stats["positive_hist"].shape[:2] == (len(rows), 4)
