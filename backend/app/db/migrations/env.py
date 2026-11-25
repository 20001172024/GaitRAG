"""Alembic environment: runs migrations against settings.database_url."""

from logging.config import fileConfig

from alembic import context

from app.db.models import Base
from app.db.session import get_engine

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_online() -> None:
    """Run migrations with a live connection (the only mode we use)."""
    url = config.attributes.get("database_url")  # set by tests; else settings
    with get_engine(url).connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
