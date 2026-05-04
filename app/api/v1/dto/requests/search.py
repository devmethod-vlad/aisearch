from pydantic import BaseModel


class SearchRequest(BaseModel):
    """ДТО поискового запроса с token-фильтрами по metadata-полям.

    Поддерживает пользовательские фильтры по raw-полям `role`, `product`
    и `component`. Значения принимаются как массивы строк и на runtime
    нормализуются оркестратором в token-представление для OpenSearch/Milvus.
    """

    query: str
    top_k: int = 5
    role: list[str] | None = None
    product: list[str] | None = None
    component: list[str] | None = None
