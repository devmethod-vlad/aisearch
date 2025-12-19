from app.common.filters.filters import (
    BaseFilter,
    BooleanFilter,
    DateFilter,
    JSONBFilter,
    NumberFilter,
    StringFilter,
    UUIDFilter,
)


class SearchRequestFilter(BaseFilter):
    """Фильтр search_request"""

    id: UUIDFilter | None = None
    query: StringFilter | None = None
    search_start_time: DateFilter | None = None
    full_execution_time: NumberFilter | None = None
    search_execution_time: NumberFilter | None = None
    dense_search_time: NumberFilter | None = None
    lex_search_time: NumberFilter | None = None
    query_norm_time: NumberFilter | None = None
    reranker_time: NumberFilter | None = None
    model_name: StringFilter | None = None
    reranker_name: StringFilter | None = None
    reranker_enable: BooleanFilter | None = None
    lex_enable: BooleanFilter | None = None
    from_cache: BooleanFilter | None = None
    lex_candidate: StringFilter | None = None
    dense_top_k: NumberFilter | None = None
    lex_top_k: NumberFilter | None = None
    top_k: NumberFilter | None = None
    weight_ce: NumberFilter | None = None
    weight_dense: NumberFilter | None = None
    weight_lex: NumberFilter | None = None
    results: JSONBFilter | None = None
    created_at: DateFilter | None = None
    modified_at: DateFilter | None = None
