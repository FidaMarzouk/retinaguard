"""Prediction records and audit log.

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("prediction_id", sa.String(36), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("input_sha256", sa.String(64), nullable=False),
        sa.Column("site", sa.String(64)),
        sa.Column("model_versions", sa.JSON, nullable=False),
        sa.Column("quality_probs", sa.JSON),
        sa.Column("quality_label", sa.String(16)),
        sa.Column("dr_probs", sa.JSON),
        sa.Column("p_referable_calibrated", sa.Float),
        sa.Column("ood_score", sa.Float),
        sa.Column("decision", sa.String(16), nullable=False),
        sa.Column("deferral_reason", sa.String(32)),
        sa.Column("latency_ms", sa.Float, nullable=False),
        sa.Column("config_hash", sa.String(64), nullable=False),
        sa.Column("git_sha", sa.String(40), nullable=False),
        sa.Column("error", sa.String(500)),
    )
    op.create_index("ix_predictions_created_at", "predictions", ["created_at"])
    op.create_index("ix_predictions_input_sha256", "predictions", ["input_sha256"])
    op.create_index("ix_predictions_decision", "predictions", ["decision"])
    op.create_table(
        "audit_log",
        sa.Column("event_id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("event", sa.String(32), nullable=False),
        sa.Column("prediction_id", sa.String(36), sa.ForeignKey("predictions.prediction_id")),
        sa.Column("detail", sa.JSON),
    )


def downgrade() -> None:
    op.drop_table("audit_log")
    op.drop_index("ix_predictions_decision", "predictions")
    op.drop_index("ix_predictions_input_sha256", "predictions")
    op.drop_index("ix_predictions_created_at", "predictions")
    op.drop_table("predictions")
