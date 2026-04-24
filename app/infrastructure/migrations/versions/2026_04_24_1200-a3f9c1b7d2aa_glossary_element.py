"""glossary_element

Revision ID: a3f9c1b7d2aa
Revises: 5bded9984796
Create Date: 2026-04-24 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "a3f9c1b7d2aa"
down_revision: Union[str, None] = "5bded9984796"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE OR REPLACE FUNCTION modified_trigger() RETURNS trigger
            LANGUAGE plpgsql
            AS $$
            BEGIN
                NEW.modified_at := NOW();
                RETURN NEW;
            END;
            $$;
        """
    )

    op.create_table(
        "glossary_element",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "abbreviation", sa.String(length=500), server_default=sa.text("''"), nullable=False
        ),
        sa.Column("term", sa.Text(), server_default=sa.text("''"), nullable=False),
        sa.Column("definition", sa.Text(), server_default=sa.text("''"), nullable=False),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "modified_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_abbreviation_trgm",
        "glossary_element",
        ["abbreviation"],
        unique=False,
        postgresql_using="gin",
        postgresql_ops={"abbreviation": "gin_trgm_ops"},
    )
    op.create_index(
        "ix_term_trgm",
        "glossary_element",
        ["term"],
        unique=False,
        postgresql_using="gin",
        postgresql_ops={"term": "gin_trgm_ops"},
    )
    op.create_index(
        "ix_definition_trgm",
        "glossary_element",
        ["definition"],
        unique=False,
        postgresql_using="gin",
        postgresql_ops={"definition": "gin_trgm_ops"},
    )

    op.execute(
        """
        CREATE TRIGGER modified_trigger
        BEFORE UPDATE ON glossary_element
        FOR EACH ROW EXECUTE FUNCTION modified_trigger();
        """
    )

    op.execute(
        """
        CREATE MATERIALIZED VIEW ge_splitted AS
        SELECT *
        FROM glossary_element
        CROSS JOIN LATERAL
            regexp_split_to_table(glossary_element.abbreviation, '; *')
            AS abbreviation_splitted
        CROSS JOIN LATERAL
            regexp_split_to_table(glossary_element.term, '; *')
            AS term_splitted;
        """
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION refresh_glossary_view() RETURNS TRIGGER
        LANGUAGE plpgsql
        AS $$
        BEGIN
            REFRESH MATERIALIZED VIEW ge_splitted;
            RETURN NULL;
        END
        $$;
        """
    )

    op.execute(
        """
        CREATE TRIGGER trig_refresh_glossary
        AFTER INSERT OR DELETE OR UPDATE ON glossary_element
        FOR EACH STATEMENT EXECUTE FUNCTION refresh_glossary_view();
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trig_refresh_glossary ON glossary_element")
    op.execute("DROP FUNCTION IF EXISTS refresh_glossary_view")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS ge_splitted")
    op.execute("DROP TRIGGER IF EXISTS modified_trigger ON glossary_element")
    op.drop_index("ix_definition_trgm", table_name="glossary_element")
    op.drop_index("ix_term_trgm", table_name="glossary_element")
    op.drop_index("ix_abbreviation_trgm", table_name="glossary_element")
    op.drop_table("glossary_element")
