import uuid

from sqlalchemy import Boolean, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.schema import ForeignKey, Index, UniqueConstraint

from app.infrastructure.models.base import Base


class SearchEntry(Base):
    __tablename__ = "search_entry"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("source.id", ondelete="RESTRICT"),
        nullable=False,
    )
    knowledge_number: Mapped[str] = mapped_column(
        String(500), nullable=False, default=""
    )
    page_id: Mapped[str] = mapped_column(String(500), nullable=False, default="")

    is_actual: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_second_line: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    role_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("role.id", ondelete="RESTRICT"),
        nullable=False,
    )
    product_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("product.id", ondelete="RESTRICT"),
        nullable=False,
    )
    component_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("component.id", ondelete="RESTRICT"),
        nullable=False,
    )

    question_markdown: Mapped[str] = mapped_column(Text, nullable=False, default="")
    question_clean: Mapped[str] = mapped_column(Text, nullable=False, default="")

    error_analysis_markdown: Mapped[str] = mapped_column(
        Text, nullable=False, default=""
    )
    error_analysis_clean: Mapped[str] = mapped_column(Text, nullable=False, default="")

    answer_markdown: Mapped[str] = mapped_column(Text, nullable=False, default="")
    answer_clean: Mapped[str] = mapped_column(Text, nullable=False, default="")

    is_for_user: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    jira: Mapped[str] = mapped_column(String(2000), nullable=False, default="")

    __table_args__ = (
        UniqueConstraint(
            "source_id",
            "knowledge_number",
            "page_id",
            name="unique_search_entry_source_knowledge_page",
        ),
        Index(
            "idx_search_entry_created_at_asc",
            "created_at",
            postgresql_using="btree",
            postgresql_ops={"created_at": "ASC"},
        ),
        Index(
            "idx_search_entry_created_at_desc",
            "created_at",
            postgresql_using="btree",
            postgresql_ops={"created_at": "DESC"},
        ),
        Index(
            "idx_search_entry_modified_at_asc",
            "modified_at",
            postgresql_using="btree",
            postgresql_ops={"modified_at": "ASC"},
        ),
        Index(
            "idx_search_entry_modified_at_desc",
            "modified_at",
            postgresql_using="btree",
            postgresql_ops={"modified_at": "DESC"},
        ),
        Index(
            "idx_search_entry_source_id",
            "source_id",
            postgresql_using="btree",
        ),
        Index(
            "idx_search_entry_role_id",
            "role_id",
            postgresql_using="btree",
        ),
        Index(
            "idx_search_entry_product_id",
            "product_id",
            postgresql_using="btree",
        ),
        Index(
            "idx_search_entry_component_id",
            "component_id",
            postgresql_using="btree",
        ),
        Index(
            "idx_search_entry_knowledge_number",
            "knowledge_number",
            postgresql_using="btree",
        ),
        Index(
            "idx_search_entry_page_id",
            "page_id",
            postgresql_using="btree",
        ),
        Index(
            "idx_search_entry_jira",
            "jira",
            postgresql_using="btree",
        ),
        Index(
            "idx_search_entry_knowledge_number_gin_trgm",
            "knowledge_number",
            postgresql_using="gin",
            postgresql_ops={"knowledge_number": "gin_trgm_ops"},
        ),
        Index(
            "idx_search_entry_page_id_gin_trgm",
            "page_id",
            postgresql_using="gin",
            postgresql_ops={"page_id": "gin_trgm_ops"},
        ),
        Index(
            "idx_search_entry_jira_gin_trgm",
            "jira",
            postgresql_using="gin",
            postgresql_ops={"jira": "gin_trgm_ops"},
        ),
        Index(
            "idx_search_entry_question_markdown_gin_trgm",
            "question_markdown",
            postgresql_using="gin",
            postgresql_ops={"question_markdown": "gin_trgm_ops"},
        ),
        Index(
            "idx_search_entry_question_clean_gin_trgm",
            "question_clean",
            postgresql_using="gin",
            postgresql_ops={"question_clean": "gin_trgm_ops"},
        ),
        Index(
            "idx_search_entry_error_analysis_markdown_gin_trgm",
            "error_analysis_markdown",
            postgresql_using="gin",
            postgresql_ops={"error_analysis_markdown": "gin_trgm_ops"},
        ),
        Index(
            "idx_search_entry_error_analysis_clean_gin_trgm",
            "error_analysis_clean",
            postgresql_using="gin",
            postgresql_ops={"error_analysis_clean": "gin_trgm_ops"},
        ),
        Index(
            "idx_search_entry_answer_markdown_gin_trgm",
            "answer_markdown",
            postgresql_using="gin",
            postgresql_ops={"answer_markdown": "gin_trgm_ops"},
        ),
        Index(
            "idx_search_entry_answer_clean_gin_trgm",
            "answer_clean",
            postgresql_using="gin",
            postgresql_ops={"answer_clean": "gin_trgm_ops"},
        ),
        Index(
            "idx_search_entry_is_actual",
            "is_actual",
            postgresql_using="btree",
        ),
        Index(
            "idx_search_entry_is_second_line",
            "is_second_line",
            postgresql_using="btree",
        ),
        Index(
            "idx_search_entry_is_for_user",
            "is_for_user",
            postgresql_using="btree",
        ),
        Index(
            "idx_search_entry_id_source_id",
            "id",
            "source_id",
            postgresql_using="btree",
        ),
        Index(
            "idx_search_entry_id_role_id",
            "id",
            "role_id",
            postgresql_using="btree",
        ),
        Index(
            "idx_search_entry_id_product_id",
            "id",
            "product_id",
            postgresql_using="btree",
        ),
        Index(
            "idx_search_entry_id_component_id",
            "id",
            "component_id",
            postgresql_using="btree",
        ),
        Index(
            "idx_search_entry_source_id_knowledge_number",
            "source_id",
            "knowledge_number",
            postgresql_using="btree",
        ),
        Index(
            "idx_search_entry_source_id_knowledge_number_page_id",
            "source_id",
            "knowledge_number",
            "page_id",
            postgresql_using="btree",
        ),
        Index(
            "idx_search_entry_role_id_product_id_component_id",
            "role_id",
            "product_id",
            "component_id",
            postgresql_using="btree",
        ),
    )
