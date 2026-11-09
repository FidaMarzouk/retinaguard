"""Prediction records and an audit log, in Postgres (Docker Compose) or SQLite (local).

Only a SHA-256 of each uploaded image is kept, never the image. The schema is created by the
Alembic migrations in ``migrations/``; ``create_schema`` builds the same tables directly, for
tests and a fresh local SQLite file.
"""

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Engine,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    func,
    insert,
    select,
    text,
)

metadata = MetaData()

predictions = Table(
    "predictions",
    metadata,
    Column("prediction_id", String(36), primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
    Column("input_sha256", String(64), nullable=False, index=True),
    Column("site", String(64)),
    Column("model_versions", JSON, nullable=False),
    Column("quality_probs", JSON),
    Column("quality_label", String(16)),
    Column("dr_probs", JSON),
    Column("p_referable_calibrated", Float),
    Column("ood_score", Float),
    Column("decision", String(16), nullable=False, index=True),
    Column("deferral_reason", String(32)),
    Column("latency_ms", Float, nullable=False),
    Column("config_hash", String(64), nullable=False),
    Column("git_sha", String(40), nullable=False),
    Column("error", String(500)),
)

audit_log = Table(
    "audit_log",
    metadata,
    Column("event_id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("event", String(32), nullable=False),
    Column("prediction_id", String(36), ForeignKey("predictions.prediction_id")),
    Column("detail", JSON),
)


def create_schema(engine: Engine) -> None:
    metadata.create_all(engine)


class PredictionStore:
    def __init__(self, url: str) -> None:
        self.engine = create_engine(url, pool_pre_ping=True)

    def healthy(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def add(self, record: Mapping[str, Any]) -> str:
        """Insert a prediction (and a ``prediction`` audit event); returns its id."""
        prediction_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        row = {**record, "prediction_id": prediction_id, "created_at": now}
        with self.engine.begin() as connection:
            connection.execute(insert(predictions).values(**row))
            connection.execute(
                insert(audit_log).values(
                    created_at=now,
                    event="prediction",
                    prediction_id=prediction_id,
                    detail={"decision": record["decision"]},
                )
            )
        return prediction_id

    def audit(
        self, event: str, detail: Mapping[str, Any], prediction_id: str | None = None
    ) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                insert(audit_log).values(
                    created_at=datetime.now(UTC),
                    event=event,
                    prediction_id=prediction_id,
                    detail=dict(detail),
                )
            )

    def get(self, prediction_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(predictions).where(predictions.c.prediction_id == prediction_id)
            ).first()
        return None if row is None else dict(row._mapping)

    def recent(self, decision: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Most recent predictions first, optionally with one decision only."""
        query = select(predictions).order_by(predictions.c.created_at.desc()).limit(limit)
        if decision is not None:
            query = query.where(predictions.c.decision == decision)
        with self.engine.connect() as connection:
            return [dict(row._mapping) for row in connection.execute(query)]

    def counts(self) -> dict[str, int]:
        """Number of predictions per decision."""
        query = select(predictions.c.decision, func.count()).group_by(predictions.c.decision)
        with self.engine.connect() as connection:
            return {str(decision): int(n) for decision, n in connection.execute(query)}

    def events(self, prediction_id: str) -> list[dict[str, Any]]:
        query = (
            select(audit_log)
            .where(audit_log.c.prediction_id == prediction_id)
            .order_by(audit_log.c.event_id)
        )
        with self.engine.connect() as connection:
            return [dict(row._mapping) for row in connection.execute(query)]
