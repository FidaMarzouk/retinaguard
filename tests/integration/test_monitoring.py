import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import dagster as dg
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from retinaguard.monitoring.batches import batch_reports, build_reference, report
from retinaguard.pipeline.monitoring import MonitoringConfig, drift_reports
from retinaguard.serving.api import create_app
from retinaguard.serving.store import PredictionStore, create_schema

pytestmark = pytest.mark.integration


def _reference() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    n = 400
    decisions = rng.choice(
        ["REFER", "NO_REFER", "DEFER_TO_HUMAN", "RETAKE"], n, p=[0.15, 0.5, 0.25, 0.1]
    )
    scored = pd.DataFrame(
        {"p_referable": rng.beta(2, 5, n), "ood_score": rng.normal(12, 2, n), "decision": decisions}
    )
    return build_reference(scored, ood_threshold=17.0)


def _record(site: str, decision: str, p: float, ood: float) -> dict[str, Any]:
    return {
        "input_sha256": "0" * 64,
        "site": site,
        "model_versions": {"grader": "x"},
        "quality_probs": None,
        "quality_label": "good",
        "dr_probs": None,
        "p_referable_calibrated": p,
        "ood_score": ood,
        "decision": decision,
        "deferral_reason": "uncertain" if decision == "DEFER_TO_HUMAN" else None,
        "latency_ms": 100.0,
        "config_hash": "c",
        "git_sha": "g",
        "error": None,
    }


def _store(tmp_path: Path) -> tuple[str, PredictionStore]:
    """30 predictions of a normal clinic and 30 of one whose photos defer and look unusual."""
    url = f"sqlite:///{(tmp_path / 'db.sqlite').as_posix()}"
    store = PredictionStore(url)
    create_schema(store.engine)
    rng = np.random.default_rng(1)
    for _ in range(30):
        decision = str(rng.choice(["REFER", "NO_REFER", "DEFER_TO_HUMAN"], p=[0.2, 0.6, 0.2]))
        store.add(_record("clinic-a", decision, float(rng.beta(2, 5)), float(rng.normal(12, 2))))
    for _ in range(30):
        p, ood = float(rng.uniform(0.1, 0.3)), float(rng.normal(25, 2))
        store.add(_record("clinic-b", "DEFER_TO_HUMAN", p, ood))
    return url, store


def test_reports_alert_and_upsert(tmp_path: Path) -> None:
    url, store = _store(tmp_path)
    reference = _reference()
    html_dir = tmp_path / "drift_reports"
    reports = report(store, reference, html_dir).set_index("site")
    assert reports.loc["clinic-a", "alert"] is None
    assert "defer_rate" in reports.loc["clinic-b", "alert"]
    assert "ood_rate" in reports.loc["clinic-b", "alert"]
    assert reports.loc["clinic-b", "psi_ood_score"] > 1
    assert reports.loc["clinic-b", "evidently_drift_share"] == 1.0
    assert reports.loc["clinic-a", "evidently_drift_share"] < 0.5
    report(store, reference)  # rerunning replaces, never duplicates
    assert len(store.drift()) == 2
    with TestClient(create_app(tmp_path / "no-models", url, drift_html_dir=html_dir)) as client:
        served = {r["site"]: r for r in client.get("/drift").json()}
        assert set(served) == {"clinic-a", "clinic-b"}
        page = client.get(served["clinic-b"]["report_url"])
        assert page.status_code == 200
        assert page.headers["content-type"].startswith("text/html")
        assert client.get("/drift/reports/..%2Fdb.sqlite").status_code == 404
        assert client.get("/drift/reports/missing.html").status_code == 404


def test_small_batches_never_alert() -> None:
    now = datetime.now(UTC)
    records = pd.DataFrame(
        [_record("tiny", "DEFER_TO_HUMAN", 0.2, 30.0) | {"created_at": now} for _ in range(5)]
    )
    reports = batch_reports(records, _reference())
    assert reports.loc[0, "alert"] is None
    assert batch_reports(pd.DataFrame(), _reference()).empty


def test_dagster_drift_asset(tmp_path: Path) -> None:
    url, _ = _store(tmp_path)
    (tmp_path / "reference.json").write_text(json.dumps(_reference()))
    config = MonitoringConfig(
        database_url=url,
        reference=str(tmp_path / "reference.json"),
        html_dir=str(tmp_path / "drift_reports"),
    )
    result = dg.materialize([drift_reports], resources={"monitoring": config})
    assert result.success
    metadata = result.asset_materializations_for_node("drift_reports")[0].metadata
    assert metadata["alerts"].value == 1
    assert len(list((tmp_path / "drift_reports").glob("drift_*.html"))) == 2
