import json
from pathlib import Path

import pytest

from retinaguard.tracking import RunLogger, default_output_dir, is_kaggle


def test_metrics_append_and_survive_reopen(tmp_path: Path) -> None:
    RunLogger(tmp_path).log_metrics({"loss": 1.5, "auroc": 0.7}, step=0)
    # A resumed session reopens the same directory and keeps earlier epochs.
    resumed = RunLogger(tmp_path)
    resumed.log_metric("loss", 1.2, step=1)

    records = resumed.read_metrics()
    assert [(r["key"], r["value"], r["step"]) for r in records] == [
        ("loss", 1.5, 0),
        ("auroc", 0.7, 0),
        ("loss", 1.2, 1),
    ]
    assert all(isinstance(r["timestamp"], int) for r in records)


def test_params_and_tags_merge(tmp_path: Path) -> None:
    logger = RunLogger(tmp_path)
    logger.log_params({"lr": 1e-4, "img_size": 512})
    logger.log_params({"lr": 3e-4})
    logger.set_tags({"model": "convnext_tiny"})

    assert json.loads((tmp_path / "params.json").read_text()) == {"img_size": 512, "lr": 3e-4}
    assert json.loads((tmp_path / "tags.json").read_text()) == {"model": "convnext_tiny"}


def test_read_metrics_empty_run(tmp_path: Path) -> None:
    assert RunLogger(tmp_path / "new").read_metrics() == []


def test_output_dir_on_kaggle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    assert is_kaggle()
    assert default_output_dir() == Path("/kaggle/working")


def test_output_dir_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    assert not is_kaggle()
    assert default_output_dir() == Path("outputs")
