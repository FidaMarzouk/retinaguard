"""FastAPI service: triage predictions, their records, health and model versions.

    uvicorn retinaguard.serving.api:app --port 8000      (or ``make serve``)

Configuration (environment): ``MODELS_DIR`` (``onnx_export`` output, default
``outputs/models``), ``DATABASE_URL`` (default a SQLite file ``outputs/retinaguard.db``; a
Postgres database must be migrated first with ``alembic upgrade head``), ``DRIFT_HTML_DIR``
(Evidently reports written by ``retinaguard.monitoring.batches``, default
``outputs/drift_reports``) and ``GIT_SHA``. Without
models the service still answers ``/health`` and record queries, and refuses predictions (503)
rather than guess. Uploaded images are never stored, only their SHA-256.
"""

import base64
import hashlib
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from retinaguard import DISCLAIMER, __version__
from retinaguard.monitoring.drift import report_filename
from retinaguard.serving.inference import Pipeline, Prediction
from retinaguard.serving.policy import Decision
from retinaguard.serving.store import PredictionStore, create_schema

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_BATCH = 32
DEFAULT_MODELS_DIR = Path("outputs/models")
DEFAULT_DATABASE_URL = "sqlite:///outputs/retinaguard.db"
DEFAULT_DRIFT_HTML_DIR = Path("outputs/drift_reports")


class Health(BaseModel):
    status: str
    version: str
    models_loaded: bool
    database: bool


class ModelInfo(BaseModel):
    versions: dict[str, str]
    lesions_available: bool
    thresholds: dict[str, Any]
    quality_classes: list[str]


class PredictionOut(BaseModel):
    prediction_id: str
    decision: str
    reason: str | None
    p_referable: float | None
    predicted_grade: int | None
    grade_probs: list[float] | None
    quality_label: str | None
    quality_probs: dict[str, float] | None
    ood_score: float | None
    lesions: dict[str, float] | None
    overlay_png_base64: str | None
    latency_ms: float
    model_versions: dict[str, str]
    disclaimer: str = DISCLAIMER


class PredictionRecord(BaseModel):
    prediction_id: str
    created_at: datetime
    input_sha256: str
    site: str | None
    model_versions: dict[str, str]
    quality_probs: dict[str, float] | None
    quality_label: str | None
    dr_probs: list[float] | None
    p_referable_calibrated: float | None
    ood_score: float | None
    decision: str
    deferral_reason: str | None
    latency_ms: float
    config_hash: str
    git_sha: str
    error: str | None


class DriftReport(BaseModel):
    site: str
    day: str
    predictions: int
    retake_rate: float | None
    defer_rate: float | None
    refer_rate: float | None
    ood_rate: float | None
    error_rate: float | None
    psi_p_referable: float | None
    psi_ood_score: float | None
    evidently_drift_share: float | None = None
    report_url: str | None = None
    alert: str | None
    created_at: datetime


def _config_hash(models_dir: Path) -> str:
    """SHA-256 of the files that define a decision: model list and fitted policy."""
    digest = hashlib.sha256()
    for name in ("models.json", "policy.json"):
        path = models_dir / name
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def create_app(
    models_dir: Path,
    database_url: str,
    git_sha: str = "unknown",
    drift_html_dir: Path = DEFAULT_DRIFT_HTML_DIR,
) -> FastAPI:
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if database_url.startswith("sqlite:///"):
            Path(database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        store = PredictionStore(database_url)
        if database_url.startswith("sqlite"):
            create_schema(store.engine)
        state["store"] = store
        try:
            state["pipeline"] = Pipeline.load(models_dir)
            store.audit("models_loaded", {"versions": state["pipeline"].versions()})
        except Exception as error:
            state["pipeline"] = None
            store.audit("models_unavailable", {"error": repr(error)[:500]})
        state["config_hash"] = _config_hash(models_dir)
        yield
        store.engine.dispose()

    app = FastAPI(
        title="RetinaGuard API", version=__version__, description=DISCLAIMER, lifespan=lifespan
    )

    def pipeline() -> Pipeline:
        loaded: Pipeline | None = state.get("pipeline")
        if loaded is None:
            raise HTTPException(503, "models are not loaded; no prediction is made")
        return loaded

    def store() -> PredictionStore:
        loaded: PredictionStore = state["store"]
        return loaded

    async def predict_one(upload: UploadFile, site: str | None, lesions: bool) -> PredictionOut:
        model = pipeline()
        data = await upload.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"{upload.filename}: image larger than 20 MB")
        if not data:
            raise HTTPException(400, f"{upload.filename}: empty upload")
        prediction: Prediction = model.predict(data, lesions=lesions)
        latency = sum(prediction.timings_ms.values())
        versions = model.versions()
        prediction_id = store().add(
            {
                "input_sha256": hashlib.sha256(data).hexdigest(),
                "site": site,
                "model_versions": versions,
                "quality_probs": prediction.quality_probs,
                "quality_label": prediction.quality_label,
                "dr_probs": prediction.grade_probs,
                "p_referable_calibrated": prediction.p_referable,
                "ood_score": prediction.ood_score,
                "decision": prediction.decision.value,
                "deferral_reason": None if prediction.reason is None else prediction.reason.value,
                "latency_ms": latency,
                "config_hash": state["config_hash"],
                "git_sha": git_sha,
                "error": None if prediction.error is None else prediction.error[:500],
            }
        )
        overlay = prediction.overlay_png
        return PredictionOut(
            prediction_id=prediction_id,
            decision=prediction.decision.value,
            reason=None if prediction.reason is None else prediction.reason.value,
            p_referable=prediction.p_referable,
            predicted_grade=prediction.predicted_grade,
            grade_probs=prediction.grade_probs,
            quality_label=prediction.quality_label,
            quality_probs=prediction.quality_probs,
            ood_score=prediction.ood_score,
            lesions=prediction.lesions,
            overlay_png_base64=None if overlay is None else base64.b64encode(overlay).decode(),
            latency_ms=latency,
            model_versions=versions,
        )

    @app.get("/health")
    def health() -> Health:
        return Health(
            status="ok",
            version=__version__,
            models_loaded=state.get("pipeline") is not None,
            database=store().healthy(),
        )

    @app.get("/models")
    def models() -> ModelInfo:
        model = pipeline()
        return ModelInfo(
            versions=model.versions(),
            lesions_available=model.has_segmentation,
            thresholds=model.thresholds.to_dict(),
            quality_classes=list(model.models["quality_classes"]),
        )

    @app.post("/predict")
    async def predict(
        image: Annotated[UploadFile, File(description="fundus photo, JPEG or PNG")],
        site: Annotated[str | None, Form()] = None,
        lesions: Annotated[bool, Form()] = False,
    ) -> PredictionOut:
        return await predict_one(image, site, lesions)

    @app.post("/predict/batch")
    async def predict_batch(
        images: Annotated[list[UploadFile], File(description="fundus photos")],
        site: Annotated[str | None, Form()] = None,
    ) -> list[PredictionOut]:
        if len(images) > MAX_BATCH:
            raise HTTPException(413, f"at most {MAX_BATCH} images per batch")
        return [await predict_one(image, site, lesions=False) for image in images]

    @app.get("/predictions")
    def list_predictions(
        decision: Decision | None = None, limit: Annotated[int, Query(ge=1, le=500)] = 50
    ) -> list[PredictionRecord]:
        rows = store().recent(None if decision is None else decision.value, limit)
        return [PredictionRecord(**row) for row in rows]

    @app.get("/drift")
    def drift(limit: Annotated[int, Query(ge=1, le=1000)] = 100) -> list[DriftReport]:
        """Drift reports per site and day (written by ``retinaguard.monitoring.batches``), with
        the address of the batch's Evidently report when one was written."""
        reports = []
        for row in store().drift(limit):
            name = report_filename(row["site"], row["day"])
            url = f"/drift/reports/{name}" if (drift_html_dir / name).is_file() else None
            reports.append(DriftReport(**row, report_url=url))
        return reports

    @app.get("/drift/reports/{name}")
    def drift_report(name: str) -> FileResponse:
        """One Evidently HTML report."""
        path = drift_html_dir / name
        if Path(name).name != name or not name.endswith(".html") or not path.is_file():
            raise HTTPException(404, "no such report")
        return FileResponse(path, media_type="text/html")

    @app.get("/predictions/{prediction_id}")
    def get_prediction(prediction_id: str) -> PredictionRecord:
        row = store().get(prediction_id)
        if row is None:
            raise HTTPException(404, "no such prediction")
        return PredictionRecord(**row)

    return app


app = create_app(
    Path(os.environ.get("MODELS_DIR", DEFAULT_MODELS_DIR)),
    os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
    os.environ.get("GIT_SHA", "unknown"),
    Path(os.environ.get("DRIFT_HTML_DIR", DEFAULT_DRIFT_HTML_DIR)),
)
