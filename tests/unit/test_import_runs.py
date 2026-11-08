from pathlib import Path

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from retinaguard.tracking import RunLogger
from retinaguard.train.import_runs import (
    CHAMPION_ALIAS,
    MODEL_NAME,
    best_epoch,
    evaluation_metrics,
    import_run,
    param_values,
)


def _run_folder(root: Path, name: str) -> Path:
    logger = RunLogger(root / name)
    logger.log_params({"model.name": "convnext_tiny", "train.lr": 2e-4, "data.limit": None})
    logger.set_tags({"task": "grader"})
    for epoch, auroc in enumerate([0.90, 0.95, 0.93]):
        logger.log_metrics({"val_referable_auroc": auroc, "train_loss": 1.0 - epoch / 10}, epoch)
    (root / name / "config.yaml").write_text("seed: 0\n")
    (root / name / "best.pt").write_bytes(b"weights")
    return root / name


@pytest.fixture
def client(tmp_path: Path) -> MlflowClient:
    return MlflowClient(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")


def test_param_values_are_strings() -> None:
    values = param_values({"a": 1, "b": None, "c": [1, 2], "d": "x", "e": "y" * 600})
    assert values == {"a": "1", "b": "null", "c": "[1, 2]", "d": "x", "e": "y" * 500}


def test_best_epoch_picks_highest_auroc() -> None:
    records = [
        {"key": "val_referable_auroc", "value": v, "step": s}
        for s, v in enumerate([0.9, 0.95, 0.93])
    ]
    assert best_epoch(records) == {"best_epoch": 1, "best_val_referable_auroc": 0.95}
    assert best_epoch([]) == {}


def test_evaluation_metrics_keeps_primary_sets() -> None:
    table = pd.DataFrame(
        {"auroc": [0.95, 0.94, 0.97], "images": [100, 120, 50]},
        index=["eyepacs_test", "eyepacs_test_all", "ddr"],
    )
    assert evaluation_metrics(table) == {
        "test.eyepacs_test.auroc": 0.95,
        "test.eyepacs_test.images": 100.0,
        "test.ddr.auroc": 0.97,
        "test.ddr.images": 50.0,
    }


def test_import_run_logs_and_registers(tmp_path: Path, client: MlflowClient) -> None:
    run_dir = _run_folder(tmp_path / "runs", "grader-convnext_tiny")
    predictions = tmp_path / "predictions" / "grader-convnext_tiny"
    predictions.mkdir(parents=True)
    pd.DataFrame({"auroc": [0.95]}, index=["eyepacs_test"]).to_csv(predictions / "metrics.csv")

    run_id = import_run(client, run_dir, predictions_root=tmp_path / "predictions", register=True)

    assert run_id is not None
    run = client.get_run(run_id)
    assert run.data.params["train.lr"] == "0.0002"
    assert run.data.tags["task"] == "grader"
    assert run.data.metrics["best_val_referable_auroc"] == 0.95
    assert run.data.metrics["test.eyepacs_test.auroc"] == 0.95
    history = client.get_metric_history(run_id, "val_referable_auroc")
    assert [m.step for m in history] == [0, 1, 2]
    artifacts = {a.path for a in client.list_artifacts(run_id)}
    assert {"config.yaml", "checkpoint", "evaluation"} <= artifacts
    champion = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    assert champion.run_id == run_id


def test_import_run_skips_already_imported(tmp_path: Path, client: MlflowClient) -> None:
    run_dir = _run_folder(tmp_path / "runs", "grader-efficientnet_b3")
    assert import_run(client, run_dir) is not None
    assert import_run(client, run_dir) is None
