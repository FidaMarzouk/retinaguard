"""Drift reports per site and day.

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"


def upgrade() -> None:
    op.create_table(
        "drift_reports",
        sa.Column("report_id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("site", sa.String(64), nullable=False),
        sa.Column("day", sa.String(10), nullable=False),
        sa.Column("predictions", sa.Integer, nullable=False),
        sa.Column("retake_rate", sa.Float),
        sa.Column("defer_rate", sa.Float),
        sa.Column("refer_rate", sa.Float),
        sa.Column("ood_rate", sa.Float),
        sa.Column("error_rate", sa.Float),
        sa.Column("psi_p_referable", sa.Float),
        sa.Column("psi_ood_score", sa.Float),
        sa.Column("alert", sa.String(500)),
        sa.UniqueConstraint("site", "day", name="uq_drift_reports_site_day"),
    )


def downgrade() -> None:
    op.drop_table("drift_reports")
