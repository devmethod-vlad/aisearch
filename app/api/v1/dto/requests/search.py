from pydantic import BaseModel, field_validator


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


class SearchPresearch(BaseModel):
    """Настройки предварительного exact-match поиска."""

    field: str

    @field_validator("field")
    @classmethod
    def validate_field(cls, value: str) -> str:
        """Нормализует и валидирует поле presearch.field."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("presearch.field не может быть пустым")
        return normalized


class SearchRequest(BaseModel):
    """DTO поискового запроса.

    Параметры runtime-поведения поиска управляются телом запроса:
    - `search_use_cache` разрешает чтение ранее сохранённого кеша результата;
    - `show_intermediate_results` включает dense/lex/ce промежуточные результаты;
    - `presearch.field` включает отдельный presearch exact-match этап.

    Фильтры передаются через необязательный объект `filters`:
    - `filters.array_filters` — фильтры по мультизначным полям;
    - `filters.exact_filters` — фильтры по одиночным строковым полям.
    """

    query: str
    top_k: int = 5
    search_use_cache: bool = True
    show_intermediate_results: bool = False
    presearch: SearchPresearch | None = None
    filters: SearchFilters | None = None
