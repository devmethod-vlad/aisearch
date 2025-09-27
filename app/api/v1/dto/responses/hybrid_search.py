from pydantic import BaseModel


class SearchResult(BaseModel):
    """Информация о результате поиска"""

    ext_id: str
    question: str
    analysis: str
    answer: str
    score_dense: float = 0.0
    score_lex: float = 0.0
    score_ce: float = 0.0
    score_final: float = 0.0
    source: str = ""


class HybridSearchResponse(BaseModel):
    """Результаты гибридного поиска"""

    results: list[SearchResult]
