"""remove_search_request_weight_ce

Revision ID: c4d8f1a2b3c4
Revises: a3f9c1b7d2aa
Create Date: 2026-05-12 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "c4d8f1a2b3c4"
down_revision: Union[str, None] = "a3f9c1b7d2aa"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index(
        "ix_search_request_weight_ce",
        table_name="search_request",
        postgresql_using="btree",
    )
    op.drop_column("search_request", "weight_ce")


def downgrade() -> None:
    op.add_column(
        "search_request",
        sa.Column("weight_ce", sa.Float(), nullable=False, server_default="0"),
    )
    op.create_index(
        "ix_search_request_weight_ce",
        "search_request",
        ["weight_ce"],
        unique=False,
        postgresql_using="btree",
    )
    op.alter_column("search_request", "weight_ce", server_default=None)
