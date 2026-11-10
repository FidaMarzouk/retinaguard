"""Evidently data-drift reports of a batch against the reference, alongside ``drift``.

Evidently's ``DataDriftPreset`` tests each column with a method it picks from the column type
and sample size (e.g. a Kolmogorov-Smirnov p-value for small numerical samples, a normed
Wasserstein distance for large ones, a z-test for categorical columns), reports the share of
drifted columns, and renders an HTML report of per-column distributions. Grader embeddings enter
as their first principal components, fitted on the reference, so the per-column tests can see
them. The alert rule stays in ``drift.alerts``: Evidently's share describes a batch, it does not
raise alerts.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

# Evidently's usage telemetry is on by default; this project sends nothing anywhere.
os.environ.setdefault("DO_NOT_TRACK", "1")

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

EMBEDDING_COMPONENTS = 10


@dataclass(frozen=True)
class EvidentlyDrift:
    drifted_columns: int
    drift_share: float
    column_scores: dict[str, float]  # per column, in the units of its method
    column_methods: dict[str, str]  # e.g. "K-S p_value", "Wasserstein distance (normed)"
    html: str


def embedding_components(
    reference: npt.ArrayLike, *others: npt.ArrayLike, k: int = EMBEDDING_COMPONENTS
) -> list[pd.DataFrame]:
    """First ``k`` principal components (fitted on ``reference``) of each embedding set."""
    ref = np.asarray(reference, dtype=np.float64)
    mean = ref.mean(axis=0)
    _, _, vt = np.linalg.svd(ref - mean, full_matrices=False)
    basis = vt[:k].T
    names = [f"embedding_pc{i + 1}" for i in range(basis.shape[1])]
    return [
        pd.DataFrame((np.asarray(x, dtype=np.float64) - mean) @ basis, columns=names)
        for x in (ref, *others)
    ]


def evidently_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    numerical: list[str],
    categorical: list[str] | None = None,
) -> EvidentlyDrift:
    """Run ``DataDriftPreset`` on the given columns of both frames."""
    categorical = categorical or []
    columns = [*numerical, *categorical]
    definition = DataDefinition(numerical_columns=numerical, categorical_columns=categorical)
    snapshot = Report([DataDriftPreset()]).run(
        Dataset.from_pandas(current[columns].reset_index(drop=True), data_definition=definition),
        Dataset.from_pandas(reference[columns].reset_index(drop=True), data_definition=definition),
    )
    count, share = 0, float("nan")
    scores: dict[str, float] = {}
    methods: dict[str, str] = {}
    for metric in snapshot.dict()["metrics"]:
        config, value = metric["config"], metric["value"]
        if config["type"].endswith("DriftedColumnsCount"):
            count, share = int(value["count"]), float(value["share"])
        elif config["type"].endswith("ValueDrift"):
            scores[str(config["column"])] = float(value)
            methods[str(config["column"])] = str(config["method"])
    return EvidentlyDrift(count, share, scores, methods, snapshot.get_html_str(as_iframe=False))


def save_html(drift: EvidentlyDrift, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(drift.html, encoding="utf-8")
    return path
