"""Add paper_tables.source ("pdf" or "xml") and n_header_rows.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-25 18:18:23.012153
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "paper_tables",
        sa.Column("source", sa.String(length=8), server_default="pdf", nullable=False),
    )
    op.add_column("paper_tables", sa.Column("n_header_rows", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("paper_tables", "n_header_rows")
    op.drop_column("paper_tables", "source")
