"""Evidently's share of drifted columns in drift reports.

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"


def upgrade() -> None:
    op.add_column("drift_reports", sa.Column("evidently_drift_share", sa.Float))


def downgrade() -> None:
    op.drop_column("drift_reports", "evidently_drift_share")
