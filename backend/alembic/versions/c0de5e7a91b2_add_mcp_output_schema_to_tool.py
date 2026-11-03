"""add mcp_output_schema to tool

The output schema that an MCP server declares for a tool. Code mode types the
tool's result from it.

Revision ID: c0de5e7a91b2
Revises: 70aed6c27639
Create Date: 2026-11-02 21:00:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "c0de5e7a91b2"
down_revision = "70aed6c27639"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tool",
        sa.Column("mcp_output_schema", postgresql.JSONB(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("tool", "mcp_output_schema")
