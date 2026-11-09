"""Decision policy: from a quality label and a referable score to a triage decision.

    quality reject              -> RETAKE          (image_quality)
    OOD score above threshold   -> DEFER_TO_HUMAN  (out_of_distribution)
    p_referable < lambda_low    -> NO_REFER
    p_referable >= lambda_high  -> REFER
    otherwise                   -> DEFER_TO_HUMAN  (uncertain)

Fails safe: a missing or invalid input, or any error while deciding, gives
``DEFER_TO_HUMAN`` (reason ``error``), never ``NO_REFER``.
"""

import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class Decision(StrEnum):
    REFER = "REFER"
    NO_REFER = "NO_REFER"
    DEFER_TO_HUMAN = "DEFER_TO_HUMAN"
    RETAKE = "RETAKE"


class Reason(StrEnum):
    IMAGE_QUALITY = "image_quality"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    UNCERTAIN = "uncertain"
    ERROR = "error"


@dataclass(frozen=True)
class Thresholds:
    """Fitted policy parameters (``calibrate`` writes them as ``policy.json``)."""

    lambda_low: float
    lambda_high: float
    temperature_referable: float = 1.0
    ood_threshold: float | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.lambda_low <= self.lambda_high:
            raise ValueError(
                f"need 0 <= lambda_low <= lambda_high, got {self.lambda_low}, {self.lambda_high}"
            )
        if not self.temperature_referable > 0:
            raise ValueError(f"temperature must be positive, got {self.temperature_referable}")

    @classmethod
    def load(cls, path: Path) -> "Thresholds":
        return cls.load_dict(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def load_dict(cls, fields: dict[str, Any]) -> "Thresholds":
        """From a ``policy.json`` mapping; keys other than the fields are ignored."""
        return cls(**{name: fields[name] for name in cls.__dataclass_fields__ if name in fields})

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


@dataclass(frozen=True)
class Outcome:
    decision: Decision
    reason: Reason | None = None


def _decide(
    p_referable: float | None,
    thresholds: Thresholds,
    quality_label: str | None,
    ood_score: float | None,
) -> Outcome:
    if quality_label == "reject":
        return Outcome(Decision.RETAKE, Reason.IMAGE_QUALITY)
    if thresholds.ood_threshold is not None:
        if ood_score is None or not math.isfinite(ood_score):
            return Outcome(Decision.DEFER_TO_HUMAN, Reason.ERROR)
        if ood_score > thresholds.ood_threshold:
            return Outcome(Decision.DEFER_TO_HUMAN, Reason.OUT_OF_DISTRIBUTION)
    if p_referable is None or not 0 <= p_referable <= 1:  # also catches NaN
        return Outcome(Decision.DEFER_TO_HUMAN, Reason.ERROR)
    if p_referable < thresholds.lambda_low:
        return Outcome(Decision.NO_REFER)
    if p_referable >= thresholds.lambda_high:
        return Outcome(Decision.REFER)
    return Outcome(Decision.DEFER_TO_HUMAN, Reason.UNCERTAIN)


def decide(
    p_referable: float | None,
    thresholds: Thresholds,
    *,
    quality_label: str | None = None,
    ood_score: float | None = None,
) -> Outcome:
    """Triage decision for one image; ``p_referable`` is the calibrated referable score."""
    try:
        return _decide(p_referable, thresholds, quality_label, ood_score)
    except Exception:
        return Outcome(Decision.DEFER_TO_HUMAN, Reason.ERROR)
