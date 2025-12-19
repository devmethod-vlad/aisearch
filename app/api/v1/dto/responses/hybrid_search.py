import typing as tp

from pydantic import BaseModel


class SearchMetrics(BaseModel):
    """Метрики выполнения поиска (все значения в секундах)"""

    embedding_time: float | None = None
    vector_search_time: float | None = None
    opensearch_time: float | None = None
    cross_encoder_time: float | None = None
    total_time: float | None = None


# class SearchResult(BaseModel):
#     """Информация о результате поиска"""

#     ext_id: str
#     question: str
#     analysis: str
#     answer: str
#     score_dense: float = 0.0
#     score_lex: float = 0.0
#     score_ce: float = 0.0
#     score_final: float = 0.0
#     sources: list = []


class HybridSearchResponse(BaseModel):
    """Результаты гибридного поиска"""

    results: list[dict[str, tp.Any]]
    search_request_id: str
    metrics: dict[str, tp.Any] | None = None
    intermediate_results: dict[str, tp.Any] | None = None
