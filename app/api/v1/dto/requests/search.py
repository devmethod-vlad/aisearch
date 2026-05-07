from pydantic import BaseModel


class SearchArrayFilters(BaseModel):
    """Фильтры по мультизначным metadata-полям.

    Каждое поле принимает список строк. Runtime-нормализация выполняется
    оркестратором в token-представление для OpenSearch/Milvus.
    """

    role: list[str] | None = None
    product: list[str] | None = None
    component: list[str] | None = None


class SearchExactFilters(BaseModel):
    """Фильтры по single-value metadata-полям.

    Каждое поле принимает одну строку. Runtime-нормализация выполняется
    оркестратором перед построением фильтров OpenSearch/Milvus.
    """

    source: str | None = None
    actual: str | None = None
    second_line: str | None = None


class SearchFilters(BaseModel):
    """Группа поисковых metadata-фильтров."""

    array_filters: SearchArrayFilters | None = None
    exact_filters: SearchExactFilters | None = None


class SearchRequest(BaseModel):
    """DTO поискового запроса.

    Фильтры передаются через необязательный объект `filters`:
    - `filters.array_filters` — фильтры по мультизначным полям;
    - `filters.exact_filters` — фильтры по одиночным строковым полям.
    """

    query: str
    top_k: int = 5
    filters: SearchFilters | None = None
