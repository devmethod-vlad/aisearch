import typing as tp

from pydantic import BaseModel


class TaskInfoResponse(BaseModel):
    """Информация о статусе задачи"""

    status: str
    info: tp.Any = None


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


class TaskQueryResponse(BaseModel):
    """Информация о задаче"""

    task_id: str
    url: str
    status: tp.Optional[str] = None
    extra: tp.Optional[dict] = None
    answer: tp.Optional[str] = None
    results: tp.Optional[list[SearchResult]] = None
