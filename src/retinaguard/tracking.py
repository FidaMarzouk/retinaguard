"""Lightweight run logger for Kaggle notebooks.

Kaggle sessions cannot reach the local MLflow server, so runs are written to plain files
(``params.json``, ``metrics.jsonl``, ``tags.json``) and imported into MLflow later.
"""

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def is_kaggle() -> bool:
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def default_output_dir() -> Path:
    """``/kaggle/working`` on Kaggle (persisted with the notebook), ``./outputs`` elsewhere."""
    return Path("/kaggle/working") if is_kaggle() else Path("outputs")


class RunLogger:
    """Append-only, resume-safe logger: reopening the same run directory keeps prior metrics."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = run_dir / "metrics.jsonl"

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._merge_json(self.run_dir / "params.json", params)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        self._merge_json(self.run_dir / "tags.json", tags)

    def log_metric(self, key: str, value: float, step: int = 0) -> None:
        record = {"key": key, "value": float(value), "step": step, "timestamp": _now_ms()}
        with self._metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def log_metrics(self, metrics: Mapping[str, float], step: int = 0) -> None:
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def read_metrics(self) -> list[dict[str, Any]]:
        if not self._metrics_path.exists():
            return []
        with self._metrics_path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    @staticmethod
    def _merge_json(path: Path, values: Mapping[str, Any]) -> None:
        current: dict[str, Any] = {}
        if path.exists():
            current = json.loads(path.read_text(encoding="utf-8"))
        current.update(values)
        path.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")


def _now_ms() -> int:
    return int(time.time() * 1000)
