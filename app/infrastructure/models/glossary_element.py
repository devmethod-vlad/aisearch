import uuid

from sqlalchemy import String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.schema import Index

from app.infrastructure.models.base import Base


class GlossaryElement(Base):
    """Элемент глоссария аббревиатур, используемый для enrichment и поиска."""

    __tablename__ = "glossary_element"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    abbreviation: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    term: Mapped[str] = mapped_column(Text, nullable=False, default="")
    definition: Mapped[str] = mapped_column(Text, nullable=False, default="")

    __table_args__ = (
        Index(
            "ix_abbreviation_trgm",
            "abbreviation",
            postgresql_using="gin",
            postgresql_ops={"abbreviation": "gin_trgm_ops"},
        ),
        Index(
            "ix_term_trgm",
            "term",
            postgresql_using="gin",
            postgresql_ops={"term": "gin_trgm_ops"},
        ),
        Index(
            "ix_definition_trgm",
            "definition",
            postgresql_using="gin",
            postgresql_ops={"definition": "gin_trgm_ops"},
        ),
    )
