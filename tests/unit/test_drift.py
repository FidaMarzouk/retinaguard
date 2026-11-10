from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from retinaguard.data.preprocess import encode_jpeg
from retinaguard.monitoring.drift import (
    alerts,
    decision_rates,
    mean_shift,
    mmd,
    psi,
    report_filename,
)
from retinaguard.monitoring.evidently_reports import embedding_components, evidently_drift
from retinaguard.monitoring.replay import degrade, replay, replay_summary
from retinaguard.monitoring.simulate import batch_metrics, batches

STATS = ("brightness", "contrast", "sharpness", "dark_fraction", "bright_fraction")


def test_psi() -> None:
    rng = np.random.default_rng(0)
    reference = rng.beta(2, 5, 5000)
    assert psi(reference, rng.beta(2, 5, 2000)) < 0.02
    assert psi(reference, rng.beta(5, 2, 2000)) > 0.25
    assert np.isnan(psi(reference, []))


def test_mmd_detects_a_shift_only_when_there_is_one() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(400, 8))
    same, p_same = mmd(reference, rng.normal(size=(300, 8)), permutations=100)
    moved, p_moved = mmd(reference, rng.normal(size=(300, 8)) + 0.5, permutations=100)
    assert p_same > 0.05
    assert p_moved < 0.02
    assert moved > same


def test_mean_shift_and_rates() -> None:
    reference = pd.DataFrame({"brightness": [10.0, 20.0, 30.0]})
    shift = mean_shift(reference, pd.DataFrame({"brightness": [40.0, 40.0]}))
    assert shift["brightness"] == pytest.approx(2.0)
    rates = decision_rates(
        pd.Series(["REFER", "RETAKE", "DEFER_TO_HUMAN", "DEFER_TO_HUMAN"]),
        pd.Series([True, False, False, False]),
    )
    assert rates == {"retake_rate": 0.25, "defer_rate": 0.5, "refer_rate": 0.25, "ood_rate": 0.25}


def test_alert_rule() -> None:
    reference = {"retake_rate": 0.1, "defer_rate": 0.2, "ood_rate": 0.02}
    assert alerts({"retake_rate": 0.19, "defer_rate": 0.39, "ood_rate": 0.039}, reference) == []
    raised = alerts({"retake_rate": 0.1, "defer_rate": 0.45, "ood_rate": 0.01}, reference)
    assert [a.signal for a in raised] == ["defer_rate"]
    # degraded photos: the gate sends them back, deferrals fall, the retake rate alerts
    degraded = alerts({"retake_rate": 0.8, "defer_rate": 0.05, "ood_rate": 0.0}, reference)
    assert [a.signal for a in degraded] == ["retake_rate"]
    assert "2 x reference" in str(raised[0])
    assert alerts({"defer_rate": float("nan")}, reference) == []


def _scored(n: int, seed: int, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grade = rng.integers(0, 5, n)
    p = np.clip((grade >= 2) * 0.6 + rng.normal(0.2 + shift, 0.1, n), 0, 1)
    decision = np.where(p > 0.5, "REFER", np.where(p < 0.1, "NO_REFER", "DEFER_TO_HUMAN"))
    return pd.DataFrame(
        {
            "row": np.arange(n),
            "dr_grade": pd.array(grade, dtype="Int64"),
            "p_referable": p,
            "decision": decision,
            "ood_flag": rng.random(n) < 0.02,
            "ood_score": rng.normal(12 + 4 * shift, 2, n),
        }
        | {s: rng.normal(50 + 20 * shift, 5, n) for s in STATS}
        | {f"embedding_pc{i + 1}": rng.normal(shift, 1, n) for i in range(10)}
    )


def test_batch_metrics_and_batches() -> None:
    reference = _scored(400, 0)
    batch = _scored(200, 1, shift=0.3).assign(row=np.arange(400, 600))
    embeddings = np.random.default_rng(2).normal(size=(600, 4))
    embeddings[400:] += 1.0
    rates = decision_rates(reference["decision"], reference["ood_flag"])
    metrics = batch_metrics(batch, reference, embeddings, rates)
    assert metrics["psi_p_referable"] > 0.1
    assert metrics["mmd_p"] < 0.05
    assert metrics["shift_brightness"] > 0.5
    assert metrics["evidently_drift_share"] > 0.5
    assert 0 <= metrics["auroc"] <= 1
    sites = ("eyepacs_test", "aptos", "idrid", "ddr")
    made = batches({s: _scored(600, i) for i, s in enumerate(sites)}, 250, seed=0)
    assert [day for scenario, day, _ in made if scenario == "aptos"] == [0, 1]
    gradual = [rows for scenario, _, rows in made if scenario == "gradual"]
    assert len(gradual) == 11
    assert all(len(rows) == 250 for rows in gradual)


def test_degrade() -> None:
    img = np.full((100, 200, 3), 200, np.uint8)
    img[:, :100] = 0
    assert 0 < degrade(img, blur=2)[50, 99, 0] < 200  # the edge is smoothed
    assert degrade(img, darken=0.5)[0, 150, 0] == 100
    assert np.array_equal(degrade(img), img)


def test_replay_posts_each_photo_with_its_site(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(encode_jpeg(np.full((64, 64, 3), 120, np.uint8)))
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.content)
        body = {
            "prediction_id": "p1",
            "decision": "RETAKE",
            "reason": "image_quality",
            "p_referable": None,
            "ood_score": None,
            "quality_label": "reject",
            "latency_ms": 10.0,
        }
        return httpx.Response(200, json=body)

    table = replay(
        [tmp_path / "a.jpg"],
        "http://api",
        "site-x",
        darken=0.5,
        transport=httpx.MockTransport(handler),
    )
    assert table.loc[0, "decision"] == "RETAKE"
    assert b"site-x" in sent[0]


def test_replay_summary() -> None:
    reference = {
        "ood_threshold": 20.0,
        "rates": {"defer_rate": 0.1, "ood_rate": 0.01},
        "p_referable": list(np.linspace(0, 1, 100)),
        "ood_score": list(np.linspace(5, 15, 100)),
    }
    tables = pd.DataFrame(
        {
            "site": ["clean"] * 4 + ["blurred"] * 4,
            "decision": [
                "REFER",
                "NO_REFER",
                "REFER",
                "NO_REFER",
                "RETAKE",
                "DEFER_TO_HUMAN",
                "DEFER_TO_HUMAN",
                "NO_REFER",
            ],
            "p_referable": [0.9, 0.01, 0.8, 0.02, None, 0.2, 0.3, 0.01],
            "ood_score": [10.0, 11.0, 12.0, 9.0, None, 25.0, 12.0, 10.0],
            "dr_grade": [3, 0, 2, 0, 3, 2, 0, 2],
        }
    )
    summary = replay_summary(tables, reference).set_index("batch")
    assert summary.loc["clean", "auroc_graded"] == 1.0
    assert summary.loc["clean", "fnr"] == 0.0
    assert summary.loc["blurred", "retake_rate"] == 0.25
    assert summary.loc["blurred", "fnr"] == 0.5  # one of two graded referable eyes cleared
    assert summary.loc["blurred", "referable_retaken"] == pytest.approx(1 / 3)
    assert "defer_rate" in summary.loc["blurred", "alert"]


def test_report_filename_is_safe() -> None:
    assert report_filename("clinic a/../x", "2026-11-10") == "drift_clinic_a_.._x_2026-11-10.html"
    assert "/" not in report_filename("../../etc", "2026-11-10")


def test_evidently_drift_flags_a_shift_only() -> None:
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"p": rng.beta(2, 5, 600), "ood": rng.normal(12, 2, 600)})
    same = pd.DataFrame({"p": rng.beta(2, 5, 200), "ood": rng.normal(12, 2, 200)})
    moved = pd.DataFrame({"p": rng.beta(5, 2, 200), "ood": rng.normal(16, 2, 200)})
    calm = evidently_drift(reference, same, numerical=["p", "ood"])
    drifted = evidently_drift(reference, moved, numerical=["p", "ood"])
    assert calm.drift_share < 0.5
    assert drifted.drift_share == 1.0
    assert drifted.drifted_columns == 2
    assert set(drifted.column_methods) == {"p", "ood"}
    assert "<html" in drifted.html.lower()


def test_embedding_components() -> None:
    rng = np.random.default_rng(0)
    reference, other = embedding_components(
        rng.normal(size=(100, 20)), rng.normal(size=(7, 20)), k=3
    )
    assert list(reference.columns) == ["embedding_pc1", "embedding_pc2", "embedding_pc3"]
    assert other.shape == (7, 3)
    assert np.allclose(reference.mean(), 0, atol=1e-9)
