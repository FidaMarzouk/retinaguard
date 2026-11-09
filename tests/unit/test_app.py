import json
from pathlib import Path

import httpx
import pandas as pd
import pytest

from retinaguard.app.client import ApiClient, ApiError
from retinaguard.app.summary import decision_mix
from retinaguard.app.views import (
    OUTCOMES,
    daily_decisions,
    decision_text,
    health_stats,
    load_summary,
    records_frame,
)
from retinaguard.serving.policy import Thresholds


def _record(decision: str, reason: str | None = None, day: int = 1, p: float | None = 0.5) -> dict:
    return {
        "created_at": f"2026-11-{day:02d}T10:00:00+00:00",
        "decision": decision,
        "deferral_reason": reason,
        "site": "a",
        "p_referable_calibrated": p,
        "quality_label": "good",
        "ood_score": 10.0,
        "latency_ms": 100.0 * day,
        "prediction_id": f"id-{decision}-{day}",
        "input_sha256": "0" * 64,
    }


def test_decision_text_explains_the_reason() -> None:
    assert decision_text("REFER", None).startswith("Refer")
    text = decision_text("DEFER_TO_HUMAN", "out_of_distribution")
    assert "unlike the images" in text


def test_records_health_and_daily_counts() -> None:
    records = [
        _record("REFER", day=1),
        _record("RETAKE", "image_quality", day=1, p=None),
        _record("DEFER_TO_HUMAN", "error", day=2, p=None),
        _record("NO_REFER", day=3, p=0.01),
    ]
    frame = records_frame(records)
    assert frame["decision"].tolist()[0] == "NO_REFER"  # most recent first
    stats = health_stats(records)
    assert stats["predictions"] == 4
    assert stats["retake_rate"] == 0.25
    assert stats["error_rate"] == 0.25
    assert stats["mean_p_referable"] == pytest.approx((0.5 + 0.01) / 2)
    daily = daily_decisions(records)
    assert daily["predictions"].sum() == 4
    assert health_stats([]) == {}
    assert records_frame([]).empty


def test_client_calls_and_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/predictions":
            assert request.url.params["decision"] == "RETAKE"
            return httpx.Response(200, json=[])
        if request.url.path == "/predict":
            assert b'name="lesions"' in request.content
            return httpx.Response(200, json={"decision": "REFER"})
        return httpx.Response(503, text="models are not loaded")

    api = ApiClient("http://api", transport=httpx.MockTransport(handler))
    assert api.health() == {"status": "ok"}
    assert api.predictions("RETAKE") == []
    assert api.predict(b"jpeg", "x.jpg", site="a", lesions=True) == {"decision": "REFER"}
    with pytest.raises(ApiError, match="503"):
        api.models()


def test_decision_mix_matches_the_policy() -> None:
    rows = pd.DataFrame(
        {
            "p_referable": [0.01, 0.01, 0.5, 0.9, 0.9],
            "quality_pred": ["ok", "ok", "ok", "ok", "reject"],
            "ood_score": [1.0, 1.0, 1.0, 1.0, 1.0],
            "dr_grade": pd.array([0, 2, 2, 3, 4], dtype="Int64"),
        }
    )
    mix = decision_mix(rows, Thresholds(lambda_low=0.05, lambda_high=0.8, ood_threshold=5.0))
    assert sum(mix[o] for o in OUTCOMES) == pytest.approx(1)
    assert mix["RETAKE"] == pytest.approx(0.2)
    assert mix["fnr"] == pytest.approx(1 / 3)  # one of three graded referable eyes cleared
    assert mix["human_workload"] == pytest.approx(0.4)


def test_load_summary(tmp_path: Path) -> None:
    assert load_summary(tmp_path / "missing.json") is None
    (tmp_path / "s.json").write_text(json.dumps({"sites": []}))
    assert load_summary(tmp_path / "s.json") == {"sites": []}
