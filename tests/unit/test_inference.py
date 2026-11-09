from pathlib import Path

import cv2
import numpy as np
import pytest

from retinaguard.data.preprocess import decode_rgb, encode_jpeg
from retinaguard.serving.inference import Pipeline, Prediction
from retinaguard.serving.policy import Decision, Reason

FIXTURE = Path(__file__).parents[1] / "fixtures" / "idrid" / "IDRiD_005.jpg"  # CC BY 4.0


@pytest.fixture(scope="module")
def pipeline(models_dir: Path) -> Pipeline:
    return Pipeline.load(models_dir)


def test_predict_returns_a_decision_with_scores(pipeline: Pipeline) -> None:
    prediction = pipeline.predict(FIXTURE.read_bytes())
    assert prediction.error is None
    if prediction.decision == Decision.RETAKE:
        assert prediction.quality_label == "reject"
        return
    assert prediction.decision in {Decision.REFER, Decision.NO_REFER, Decision.DEFER_TO_HUMAN}
    assert prediction.p_referable is not None
    assert 0 <= prediction.p_referable <= 1
    assert prediction.grade_probs is not None
    assert sum(prediction.grade_probs) == pytest.approx(1)
    assert prediction.predicted_grade in range(5)
    assert prediction.ood_score is not None
    assert set(prediction.timings_ms) == {"preprocess", "quality", "grader", "policy"}


def test_predict_with_lesions_returns_an_overlay(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        pipeline.models, "quality_classes", ["good", "usable", "good"]
    )  # never reject
    prediction = pipeline.predict(FIXTURE.read_bytes(), lesions=True)
    assert prediction.lesions is not None
    assert set(prediction.lesions) == {
        "microaneurysms",
        "haemorrhages",
        "hard exudates",
        "soft exudates",
    }
    assert prediction.overlay_png is not None
    overlay = cv2.imdecode(np.frombuffer(prediction.overlay_png, np.uint8), cv2.IMREAD_COLOR)
    assert overlay.shape == (128, 128, 3)
    assert "segmentation" in prediction.timings_ms


def test_quality_reject_is_retake(pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(pipeline.models, "quality_classes", ["reject", "reject", "reject"])
    prediction = pipeline.predict(FIXTURE.read_bytes())
    assert prediction.decision == Decision.RETAKE
    assert prediction.reason == Reason.IMAGE_QUALITY
    assert prediction.p_referable is None


def test_no_fundus_is_retake(pipeline: Pipeline) -> None:
    black = encode_jpeg(np.zeros((300, 300, 3), np.uint8))
    prediction = pipeline.predict(black)
    assert prediction.decision == Decision.RETAKE
    assert prediction.reason == Reason.IMAGE_QUALITY


def test_errors_fail_safe(pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch) -> None:
    garbage = pipeline.predict(b"not an image")
    assert (garbage.decision, garbage.reason) == (Decision.DEFER_TO_HUMAN, Reason.ERROR)
    assert garbage.error is not None

    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("model crashed")

    monkeypatch.setattr(pipeline.sessions["grader"], "run", broken)
    monkeypatch.setitem(pipeline.models, "quality_classes", ["good", "good", "good"])
    crashed = pipeline.predict(FIXTURE.read_bytes())
    assert (crashed.decision, crashed.reason) == (Decision.DEFER_TO_HUMAN, Reason.ERROR)
    assert "model crashed" in str(crashed.error)


def test_load_checks_files(tmp_path: Path, models_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"models\.json"):
        Pipeline.load(tmp_path)
    pipeline = Pipeline.load(models_dir)
    assert pipeline.has_segmentation
    assert set(pipeline.versions()) == {"quality", "grader", "segmentation"}


def test_prediction_grade() -> None:
    assert Prediction(Decision.REFER, grade_probs=[0.1, 0.1, 0.6, 0.1, 0.1]).predicted_grade == 2
    assert decode_rgb(FIXTURE.read_bytes()).ndim == 3
