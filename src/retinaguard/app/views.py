"""Display logic of the app, kept free of Streamlit so it can be tested. Imports only light
dependencies: the app image has no model libraries."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DECISION_TEXT = {
    "REFER": "Refer to an eye specialist: signs of referable diabetic retinopathy.",
    "NO_REFER": "No referable retinopathy found: routine re-screening.",
    "DEFER_TO_HUMAN": "A human grader must decide.",
    "RETAKE": "Retake the photo: it cannot be graded.",
}
REASON_TEXT = {
    "uncertain": "the score lies between the clearing and referral thresholds",
    "out_of_distribution": "the photo looks unlike the images the model was trained on",
    "image_quality": "the quality check judged the photo ungradable",
    "error": "something went wrong while processing the photo",
}
# Fixed colour per decision (text stays in the theme's ink colour; this marks the badge only).
DECISION_COLOURS = {
    "REFER": "#1c5cab",
    "NO_REFER": "#86b6ef",
    "DEFER_TO_HUMAN": "#d3d1c7",
    "RETAKE": "#b4b2ab",
}
SUMMARY_PATH = Path(__file__).parent / "site_summary.json"
OUTCOMES = ("RETAKE", "DEFER (out of distribution)", "DEFER (uncertain)", "NO_REFER", "REFER")
GRADE_NAMES = ["0 none", "1 mild", "2 moderate", "3 severe", "4 proliferative"]
RECORD_COLUMNS = [
    "created_at",
    "decision",
    "deferral_reason",
    "site",
    "p_referable_calibrated",
    "quality_label",
    "ood_score",
    "latency_ms",
    "prediction_id",
]


def load_summary(path: Path = SUMMARY_PATH) -> dict[str, Any] | None:
    """Per-site results written by ``python -m retinaguard.app.summary``, if present."""
    if not path.exists():
        return None
    summary: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return summary


def decision_text(decision: str, reason: str | None) -> str:
    text = DECISION_TEXT.get(decision, decision)
    if reason:
        text += f" Reason: {REASON_TEXT.get(reason, reason.replace('_', ' '))}."
    return text


def records_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Stored predictions as a table with readable columns, most recent first."""
    if not records:
        return pd.DataFrame(columns=RECORD_COLUMNS)
    frame = pd.DataFrame(records)
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True)
    return frame[RECORD_COLUMNS].sort_values("created_at", ascending=False).reset_index(drop=True)


def health_stats(records: list[dict[str, Any]]) -> dict[str, float]:
    """Rates and latency over stored predictions (empty dict when there are none)."""
    if not records:
        return {}
    frame = records_frame(records)
    decisions = frame["decision"]
    latency = frame["latency_ms"].astype(float)
    p = frame["p_referable_calibrated"].dropna().astype(float)
    return {
        "predictions": float(len(frame)),
        "retake_rate": float((decisions == "RETAKE").mean()),
        "defer_rate": float((decisions == "DEFER_TO_HUMAN").mean()),
        "refer_rate": float((decisions == "REFER").mean()),
        "error_rate": float((frame["deferral_reason"] == "error").mean()),
        "mean_p_referable": float(p.mean()) if len(p) else float("nan"),
        "latency_p50_ms": float(np.percentile(latency, 50)),
        "latency_p95_ms": float(np.percentile(latency, 95)),
    }


def daily_decisions(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Predictions per day and decision (long format, for a stacked chart)."""
    frame = records_frame(records)
    if frame.empty:
        return pd.DataFrame(columns=["day", "decision", "predictions"])
    frame["day"] = frame["created_at"].dt.floor("D")
    counts = frame.groupby(["day", "decision"]).size().rename("predictions")
    return counts.reset_index()
