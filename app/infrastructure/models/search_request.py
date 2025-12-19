import datetime
import uuid

from sqlalchemy import (
    TIMESTAMP,
    Boolean,
    Float,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.schema import Index, UniqueConstraint

from app.infrastructure.models.base import Base


class SearchRequest(Base):
    __tablename__ = "search_request"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    query: Mapped[str] = mapped_column(Text, nullable=False, default="")
    search_start_time: Mapped[datetime.datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    full_execution_time: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    search_execution_time: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    dense_search_time: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lex_search_time: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    query_norm_time: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    reranker_time: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    model_name: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    reranker_name: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    reranker_enable: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    lex_enable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    from_cache: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    lex_candidate: Mapped[str] = mapped_column(
        String(500), nullable=True, default="OpenSearch"
    )
    dense_top_k: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lex_top_k: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    weight_ce: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    weight_dense: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    weight_lex: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    results: Mapped[dict] = mapped_column(JSONB, nullable=False, default=list)

    __table_args__ = (
        UniqueConstraint("id", name="unique_search_request_id"),
        Index(
            "ix_search_request_query_gin_trgm",
            "query",
            postgresql_using="gin",
            postgresql_ops={"query": "gin_trgm_ops"},
        ),
        Index(
            "ix_search_request_model_name_gin_trgm",
            "model_name",
            postgresql_using="gin",
            postgresql_ops={"model_name": "gin_trgm_ops"},
        ),
        Index(
            "ix_search_request_reranker_name_gin_trgm",
            "reranker_name",
            postgresql_using="gin",
            postgresql_ops={"reranker_name": "gin_trgm_ops"},
        ),
        Index(
            "ix_search_request_lex_candidate_gin_trgm",
            "lex_candidate",
            postgresql_using="gin",
            postgresql_ops={"lex_candidate": "gin_trgm_ops"},
        ),
        Index(
            "ix_search_request_results_jsonb_path",
            "results",
            postgresql_using="gin",
            postgresql_ops={"results": "jsonb_path_ops"},
        ),
        Index(
            "ix_search_request_search_start_time_asc",
            "search_start_time",
            postgresql_using="btree",
            postgresql_ops={"search_start_time": "ASC"},
        ),
        Index(
            "ix_search_request_search_start_time_desc",
            "search_start_time",
            postgresql_using="btree",
            postgresql_ops={"search_start_time": "DESC"},
        ),
        Index(
            "ix_search_request_created_at_asc",
            "created_at",
            postgresql_using="btree",
            postgresql_ops={"created_at": "ASC"},
        ),
        Index(
            "ix_search_request_created_at_desc",
            "created_at",
            postgresql_using="btree",
            postgresql_ops={"created_at": "DESC"},
        ),
        Index(
            "ix_search_request_modified_at_asc",
            "modified_at",
            postgresql_using="btree",
            postgresql_ops={"modified_at": "ASC"},
        ),
        Index(
            "ix_search_request_modified_at_desc",
            "modified_at",
            postgresql_using="btree",
            postgresql_ops={"modified_at": "DESC"},
        ),
        Index(
            "ix_search_request_full_execution_time",
            "full_execution_time",
            postgresql_using="btree",
        ),
        Index(
            "ix_search_request_search_execution_time",
            "search_execution_time",
            postgresql_using="btree",
        ),
        Index(
            "ix_search_request_dense_search_time",
            "dense_search_time",
            postgresql_using="btree",
        ),
        Index(
            "ix_search_request_lex_search_time",
            "lex_search_time",
            postgresql_using="btree",
        ),
        Index(
            "ix_search_request_query_norm_time",
            "query_norm_time",
            postgresql_using="btree",
        ),
        Index(
            "ix_search_request_reranker_time",
            "reranker_time",
            postgresql_using="btree",
        ),
        Index(
            "ix_search_request_dense_top_k",
            "dense_top_k",
            postgresql_using="btree",
        ),
        Index(
            "ix_search_request_lex_top_k",
            "lex_top_k",
            postgresql_using="btree",
        ),
        Index(
            "ix_search_request_top_k",
            "top_k",
            postgresql_using="btree",
        ),
        Index(
            "ix_search_request_weight_ce",
            "weight_ce",
            postgresql_using="btree",
        ),
        Index(
            "ix_search_request_weight_dense",
            "weight_dense",
            postgresql_using="btree",
        ),
        Index(
            "ix_search_request_weight_lex",
            "weight_lex",
            postgresql_using="btree",
        ),
        Index(
            "ix_search_request_reranker_enable",
            "reranker_enable",
            postgresql_using="btree",
        ),
        Index(
            "ix_search_request_lex_enable",
            "lex_enable",
            postgresql_using="btree",
        ),
        Index(
            "ix_search_request_from_cache",
            "from_cache",
            postgresql_using="btree",
        ),
    )
