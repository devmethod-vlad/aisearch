import uuid

from sqlalchemy import String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.schema import ForeignKey, Index, UniqueConstraint

from app.infrastructure.models.base import Base
from app.infrastructure.models.enums import RatingEnum, RatingEnumType


class SearchFeedback(Base):
    __tablename__ = "search_feedback"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    search_request_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("search_request.id", ondelete="CASCADE"),
        nullable=False,
    )
    question_id: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    rating: Mapped[RatingEnum] = mapped_column(RatingEnumType, nullable=False)
    source: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    comment: Mapped[str] = mapped_column(Text, nullable=False, default="")

    __table_args__ = (
        UniqueConstraint(
            "search_request_id",
            "question_id",
            name="unique_search_feedback_request_question",
        ),
        Index(
            "idx_search_feedback_created_at_asc",
            "created_at",
            postgresql_using="btree",
            postgresql_ops={"created_at": "ASC"},
        ),
        Index(
            "idx_search_feedback_created_at_desc",
            "created_at",
            postgresql_using="btree",
            postgresql_ops={"created_at": "DESC"},
        ),
        Index(
            "idx_search_feedback_modified_at_asc",
            "modified_at",
            postgresql_using="btree",
            postgresql_ops={"modified_at": "ASC"},
        ),
        Index(
            "idx_search_feedback_modified_at_desc",
            "modified_at",
            postgresql_using="btree",
            postgresql_ops={"modified_at": "DESC"},
        ),
        Index(
            "idx_search_feedback_search_request_id",
            "search_request_id",
            postgresql_using="btree",
        ),
        Index(
            "idx_search_feedback_question_id",
            "question_id",
            postgresql_using="btree",
        ),
        Index(
            "idx_search_feedback_source",
            "source",
            postgresql_using="btree",
        ),
        Index(
            "idx_search_feedback_comment_gin",
            "comment",
            postgresql_using="gin",
            postgresql_ops={"comment": "gin_trgm_ops"},
        ),
        Index(
            "idx_search_feedback_id_search_request_id",
            "id",
            "search_request_id",
            postgresql_using="btree",
        ),
        Index(
            "idx_search_feedback_search_request_id_question_id",
            "search_request_id",
            "question_id",
            postgresql_using="btree",
        ),
    )
