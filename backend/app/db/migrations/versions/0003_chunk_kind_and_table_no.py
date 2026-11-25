"""Add chunks.kind ("text" or "table") and chunks.table_no.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-25 18:59:32.264773
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chunks", sa.Column("kind", sa.String(length=8), server_default="text", nullable=False)
    )
    op.add_column("chunks", sa.Column("table_no", sa.String(length=16), nullable=True))


def downgrade() -> None:
    op.drop_column("chunks", "table_no")
    op.drop_column("chunks", "kind")
