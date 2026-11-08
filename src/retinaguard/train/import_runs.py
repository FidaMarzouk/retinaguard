"""Import grader run folders (written on Kaggle by ``RunLogger``) into MLflow.

    python -m retinaguard.train.import_runs outputs/runs/grader-convnext_tiny \\
        outputs/runs/grader-efficientnet_b3 --predictions-root data/interim/predictions \\
        --register grader-convnext_tiny

Each folder becomes one MLflow run named after it, with its parameters, tags, per-epoch metrics,
``config.yaml`` and ``curves.png``. If ``<predictions-root>/<run name>/metrics.csv`` exists (from
``evaluate``), the test metrics of the primary sets are logged as ``test.<set>.<metric>`` and the
table as an artifact. ``--register`` also uploads that run's ``best.pt`` and registers it as a
version of the ``dr-grader`` model with the alias ``champion``. Runs already imported are
skipped, so the command can be rerun as new runs arrive.

Tracking URI: ``--tracking-uri``, else ``MLFLOW_TRACKING_URI`` (e.g. the Docker Compose server),
else a local SQLite store in ``mlruns/`` (view with
``mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db``).
"""

import argparse
import json
import math
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient

LOCAL_STORE = Path("mlruns")
EXPERIMENT = "dr-grader"
MODEL_NAME = "dr-grader"
CHAMPION_ALIAS = "champion"
PRIMARY_SETS = ("eyepacs_test", "aptos", "idrid", "ddr")
BATCH_SIZE = 1000  # MLflow accepts at most 1000 metrics per log_batch call
PARAM_MAX_LENGTH = 500


def param_values(params: Mapping[str, Any]) -> dict[str, str]:
    """MLflow parameters are strings; lists and None are written as JSON, long values cut."""
    return {
        key: (value if isinstance(value, str) else json.dumps(value))[:PARAM_MAX_LENGTH]
        for key, value in params.items()
    }


def metric_entities(records: Iterable[Mapping[str, Any]]) -> list[Metric]:
    """``metrics.jsonl`` records as MLflow metrics, keeping step and timestamp."""
    return [
        Metric(r["key"], float(r["value"]), int(r["timestamp"]), int(r["step"])) for r in records
    ]


def best_epoch(
    records: Iterable[Mapping[str, Any]], key: str = "val_referable_auroc"
) -> dict[str, float]:
    """Summary of the selected epoch (highest ``key``), as ``train_grader`` selects it."""
    values = {int(r["step"]): float(r["value"]) for r in records if r["key"] == key}
    if not values:
        return {}
    step = max(values, key=lambda s: values[s])
    return {"best_epoch": step, f"best_{key}": values[step]}


def evaluation_metrics(table: pd.DataFrame) -> dict[str, float]:
    """``evaluate`` output rows of the primary sets, as ``test.<set>.<metric>``."""
    metrics = {}
    for name in PRIMARY_SETS:
        if name not in table.index:
            continue
        for column, value in table.loc[name].items():
            if isinstance(value, int | float) and not math.isnan(value):
                metrics[f"test.{name}.{column}"] = float(value)
    return metrics


def tracking_uri(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return os.environ["MLFLOW_TRACKING_URI"]
    LOCAL_STORE.mkdir(exist_ok=True)
    return f"sqlite:///{(LOCAL_STORE / 'mlflow.db').as_posix()}"


def experiment_id(client: MlflowClient, name: str = EXPERIMENT) -> str:
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return str(experiment.experiment_id)
    location = None
    if client.tracking_uri.startswith("sqlite:"):
        location = (LOCAL_STORE / "artifacts").resolve().as_uri()
    return str(client.create_experiment(name, artifact_location=location))


def import_run(
    client: MlflowClient,
    run_dir: Path,
    *,
    experiment: str = EXPERIMENT,
    predictions_root: Path | None = None,
    register: bool = False,
) -> str | None:
    """Import one run folder; returns the MLflow run id, or None if it was already imported."""
    name = run_dir.name
    exp_id = experiment_id(client, experiment)
    if client.search_runs([exp_id], filter_string=f"tags.`mlflow.runName` = '{name}'"):
        print(f"{name}: already imported, skipped")
        return None

    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    tags = (
        json.loads((run_dir / "tags.json").read_text()) if (run_dir / "tags.json").exists() else {}
    )
    run = client.create_run(exp_id, run_name=name, tags={**tags, "source": "kaggle run folder"})
    run_id = run.info.run_id
    params = json.loads((run_dir / "params.json").read_text())
    client.log_batch(
        run_id, params=[mlflow.entities.Param(k, v) for k, v in param_values(params).items()]
    )

    summary = best_epoch(records)
    table_path = predictions_root / name / "metrics.csv" if predictions_root else None
    if table_path is not None and table_path.exists():
        summary |= evaluation_metrics(pd.read_csv(table_path, index_col=0))
        client.log_artifact(run_id, str(table_path), "evaluation")
    end_time = max((int(r["timestamp"]) for r in records), default=None)
    entities = metric_entities(records)
    entities += [Metric(k, v, end_time or 0, 0) for k, v in summary.items()]
    for start in range(0, len(entities), BATCH_SIZE):
        client.log_batch(run_id, metrics=entities[start : start + BATCH_SIZE])

    for artifact in ("config.yaml", "curves.png"):
        if (run_dir / artifact).exists():
            client.log_artifact(run_id, str(run_dir / artifact))
    if register:
        checkpoint = run_dir / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"{checkpoint} is needed to register {name}")
        client.log_artifact(run_id, str(checkpoint), "checkpoint")
        if not client.search_registered_models(f"name = '{MODEL_NAME}'"):
            client.create_registered_model(
                MODEL_NAME, description="ICDR grade + referable DR grader"
            )
        version = client.create_model_version(
            MODEL_NAME, f"runs:/{run_id}/checkpoint", run_id=run_id
        )
        client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version.version)
        print(f"{name}: registered as {MODEL_NAME} version {version.version} ({CHAMPION_ALIAS})")
    client.set_terminated(run_id, "FINISHED", end_time=end_time)
    print(f"{name}: imported as run {run_id}")
    return str(run_id)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--predictions-root", type=Path)
    parser.add_argument("--register", help="run name whose best.pt becomes the champion model")
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--tracking-uri")
    args = parser.parse_args(argv)

    client = MlflowClient(tracking_uri(args.tracking_uri))
    for run_dir in args.run_dirs:
        import_run(
            client,
            run_dir,
            experiment=args.experiment,
            predictions_root=args.predictions_root,
            register=run_dir.name == args.register,
        )


if __name__ == "__main__":
    main()
