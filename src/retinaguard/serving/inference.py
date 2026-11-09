"""The full triage chain on one image, from exported models (no PyTorch needed).

    preprocess → quality gate → grader → OOD score → policy → optional lesion overlay

``Pipeline.load(models_dir)`` reads what ``onnx_export`` writes there: ``models.json``, the ONNX
files, and the fitted ``policy.json`` (temperatures and thresholds) and ``ood.npz``. A photo
whose fundus cannot be found is sent back (RETAKE), like a quality reject. Any other error
returns DEFER_TO_HUMAN with reason ``error``, never NO_REFER.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from retinaguard.data.preprocess import (
    FundusNotFoundError,
    Image,
    decode_rgb,
    local_mean_normalise,
    preprocess,
)
from retinaguard.models.ood import Mahalanobis
from retinaguard.serving.overlay import lesion_summary, overlay
from retinaguard.serving.policy import Decision, Outcome, Reason, Thresholds, decide

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
REQUIRED_FILES = ("models.json", "policy.json")


@dataclass
class Prediction:
    decision: Decision
    reason: Reason | None = None
    p_referable: float | None = None  # calibrated
    grade_probs: list[float] | None = None  # calibrated, ICDR 0-4
    quality_label: str | None = None
    quality_probs: dict[str, float] | None = None
    ood_score: float | None = None
    lesions: dict[str, float] | None = None  # share of the image per lesion type
    overlay_png: bytes | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    @property
    def predicted_grade(self) -> int | None:
        return None if self.grade_probs is None else int(np.argmax(self.grade_probs))


def to_batch(img: Image, local_mean_norm: bool = False) -> npt.NDArray[np.float32]:
    """(1, 3, H, W) float32 input scaled with ImageNet statistics."""
    if local_mean_norm:
        img = local_mean_normalise(img)
    scaled = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(scaled.transpose(2, 0, 1)[None])


def softmax(z: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return np.asarray(e / e.sum(), dtype=np.float64)


def sigmoid(z: float) -> float:
    return float(0.5 * (1 + np.tanh(0.5 * z)))


class Pipeline:
    def __init__(
        self,
        models: dict[str, Any],
        sessions: dict[str, ort.InferenceSession],
        policy: dict[str, Any],
        detector: Mahalanobis | None,
    ) -> None:
        self.models = models
        self.sessions = sessions
        self.policy = policy
        self.thresholds = Thresholds.load_dict(policy)
        self.detector = detector
        if self.thresholds.ood_threshold is not None and detector is None:
            raise ValueError("policy has an OOD threshold but ood.npz is missing")
        for name in ("quality", "grader"):
            if name not in sessions:
                raise ValueError(f"missing {name} model")

    @classmethod
    def load(cls, models_dir: Path, threads: int = 0) -> "Pipeline":
        for name in REQUIRED_FILES:
            if not (models_dir / name).exists():
                raise FileNotFoundError(f"{models_dir / name} not found")
        models = json.loads((models_dir / "models.json").read_text(encoding="utf-8"))
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        # Idle threads must not spin: two sessions spinning side by side slowed each model ~4x.
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        sessions = {
            name: ort.InferenceSession(
                str(models_dir / models[name]["file"]), options, providers=["CPUExecutionProvider"]
            )
            for name in ("quality", "grader", "segmentation")
            if name in models
        }
        policy = json.loads((models_dir / "policy.json").read_text(encoding="utf-8"))
        ood_path = models_dir / "ood.npz"
        detector = Mahalanobis.load(ood_path) if ood_path.exists() else None
        return cls(models, sessions, policy, detector)

    @property
    def has_segmentation(self) -> bool:
        return "segmentation" in self.sessions

    def versions(self) -> dict[str, str]:
        """Short SHA-256 of each loaded model file."""
        return {name: self.models[name]["sha256"][:12] for name in self.sessions}

    def _run(self, name: str, batch: npt.NDArray[np.float32]) -> list[npt.NDArray[np.float32]]:
        outputs: list[npt.NDArray[np.float32]] = self.sessions[name].run(None, {"images": batch})
        return outputs

    def predict(self, image_bytes: bytes, *, lesions: bool = False) -> Prediction:
        """Triage one encoded image; never raises."""
        timings: dict[str, float] = {}
        try:
            return self._predict(image_bytes, lesions, timings)
        except Exception as error:  # fail safe: an error never clears an image
            return Prediction(
                Decision.DEFER_TO_HUMAN, Reason.ERROR, timings_ms=timings, error=repr(error)
            )

    def _predict(self, image_bytes: bytes, lesions: bool, timings: dict[str, float]) -> Prediction:
        started = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal started
            now = time.perf_counter()
            timings[name] = (now - started) * 1000
            started = now

        raw = decode_rgb(image_bytes)
        size = self.models["grader"]["image_size"]
        try:
            img, _ = preprocess(raw, size)
        except FundusNotFoundError as error:
            lap("preprocess")
            return Prediction(
                Decision.RETAKE, Reason.IMAGE_QUALITY, timings_ms=timings, error=str(error)
            )
        lap("preprocess")

        quality_size = self.models["quality"]["image_size"]
        quality_img = img if quality_size == size else preprocess(raw, quality_size)[0]
        classes = self.models["quality_classes"]
        quality = softmax(self._run("quality", to_batch(quality_img))[0][0])
        quality_label = classes[int(quality.argmax())]
        quality_probs = {c: float(p) for c, p in zip(classes, quality, strict=True)}
        lap("quality")
        if quality_label == "reject":
            retake = decide(None, self.thresholds, quality_label=quality_label)
            return Prediction(
                retake.decision,
                retake.reason,
                quality_label=quality_label,
                quality_probs=quality_probs,
                timings_ms=timings,
            )

        lmn = bool(self.models["grader"]["local_mean_norm"])
        grade_logits, referable_logit, embedding = self._run("grader", to_batch(img, lmn))
        lap("grader")
        grade_probs = softmax(grade_logits[0] / self.policy.get("temperature_grade", 1.0))
        p_referable = sigmoid(float(referable_logit[0]) / self.thresholds.temperature_referable)
        ood_score = None if self.detector is None else float(self.detector.score(embedding)[0])
        outcome: Outcome = decide(
            p_referable, self.thresholds, quality_label=quality_label, ood_score=ood_score
        )
        lap("policy")
        prediction = Prediction(
            outcome.decision,
            outcome.reason,
            p_referable=p_referable,
            grade_probs=[float(p) for p in grade_probs],
            quality_label=quality_label,
            quality_probs=quality_probs,
            ood_score=ood_score,
            timings_ms=timings,
        )
        if lesions and self.has_segmentation:
            seg_size = self.models["segmentation"]["image_size"]
            seg_img = preprocess(raw, seg_size)[0]
            logits = self._run("segmentation", to_batch(seg_img))[0][0]
            probs = 1 / (1 + np.exp(-logits.astype(np.float64)))
            prediction.lesions = lesion_summary(probs)
            ok, png = cv2.imencode(".png", cv2.cvtColor(overlay(seg_img, probs), cv2.COLOR_RGB2BGR))
            prediction.overlay_png = png.tobytes() if ok else None
            lap("segmentation")
        return prediction
