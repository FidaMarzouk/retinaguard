"""Alembic environment: migrates the database named by ``DATABASE_URL``."""

import os

from alembic import context
from sqlalchemy import create_engine

from retinaguard.serving.store import metadata

url = os.environ.get("DATABASE_URL", "sqlite:///outputs/retinaguard.db")


def run_offline() -> None:
    context.configure(url=url, target_metadata=metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_online() -> None:
    engine = create_engine(url)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_offline()
else:
    run_online()
