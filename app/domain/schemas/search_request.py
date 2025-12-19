import typing as tp
import uuid
from datetime import UTC, datetime

from pydantic import Field

from app.common.arbitrary_model import ArbitraryModel


class SearchRequestCreateDTO(ArbitraryModel):
    """Схема создания результата поиска"""

    id: uuid.UUID
    query: str = ""
    search_start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    full_execution_time: int = 0
    search_execution_time: int = 0
    dense_search_time: int = 0
    lex_search_time: int = 0
    query_norm_time: int = 0
    reranker_time: int = 0
    model_name: str = ""
    reranker_name: str = ""
    reranker_enable: bool = False
    lex_enable: bool = False
    from_cache: bool = False
    lex_candidate: str = "OpenSearch"
    dense_top_k: int = 0
    lex_top_k: int = 0
    top_k: int = 0
    weight_ce: float = 0.0
    weight_dense: float = 0.0
    weight_lex: float = 0.0
    results: list[dict[str, tp.Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    modified_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SearchRequestSchema(SearchRequestCreateDTO):
    """Выходная схема результата поиска"""

    pass
