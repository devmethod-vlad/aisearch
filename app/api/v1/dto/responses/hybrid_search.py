from pydantic import BaseModel
import typing as tp


class SearchMetrics(BaseModel):
    """Метрики выполнения поиска (все значения в секундах)"""
    embedding_time: tp.Optional[float] = None
    vector_search_time: tp.Optional[float] = None
    opensearch_time: tp.Optional[float] = None
    bm25_time: tp.Optional[float] = None
    cross_encoder_time: tp.Optional[float] = None
    total_time: tp.Optional[float] = None


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
    metrics: dict[str, tp.Any] | None = None
