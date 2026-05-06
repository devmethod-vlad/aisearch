"""Тесты утилит exact-фильтрации single-value metadata полей."""

from app.infrastructure.utils.exact_filters import (
    ExactFilterConfig,
    NormalizedExactFilters,
    build_milvus_exact_filter_expr,
    build_opensearch_exact_filter_clauses,
    combine_milvus_filter_exprs,
    enrich_records_with_exact_filter_fields,
    normalize_exact_value,
    normalize_request_exact_filters,
)


def test_normalize_exact_value() -> None:
    """Проверяет trim+casefold без split."""
    assert normalize_exact_value(" Да ") == "да"
    assert normalize_exact_value("ТП") == "тп"
    assert normalize_exact_value("ЭМИАС;ЛИС") == "эмиас;лис"
    assert normalize_exact_value("") == ""


def test_exact_filter_config_fields() -> None:
    """Проверяет корректное формирование имен служебных полей."""
    cfg = ExactFilterConfig(raw_fields=("source", "actual", "second_line"), field_suffix="_filter")
    assert cfg.filter_field("source") == "source_filter"
    assert cfg.filter_field("actual") == "actual_filter"
    assert cfg.filter_field("second_line") == "second_line_filter"


def test_enrich_records_with_exact_filter_fields() -> None:
    """Проверяет enrichment без изменения raw-полей."""
    cfg = ExactFilterConfig(raw_fields=("source", "actual", "second_line"), field_suffix="_filter")
    enriched = enrich_records_with_exact_filter_fields([
        {"source": "ТП", "actual": " Да ", "second_line": "Line"}
    ], config=cfg)
    assert enriched[0]["source_filter"] == "тп"
    assert enriched[0]["actual_filter"] == "да"
    assert enriched[0]["second_line_filter"] == "line"
    assert enriched[0]["source"] == "ТП"


def test_request_and_query_builders() -> None:
    """Проверяет нормализацию фильтров и построение запросов для backend-ов."""
    cfg = ExactFilterConfig(raw_fields=("source", "actual", "second_line"), field_suffix="_filter")
    nf = normalize_request_exact_filters(
        {"source": "ТП", "actual": " Да ", "second_line": None, "unknown": "x"},
        config=cfg,
    )
    assert nf.by_filter_field == {"source_filter": "тп", "actual_filter": "да"}
    assert build_opensearch_exact_filter_clauses(nf) == [
        {"term": {"actual_filter": "да"}},
        {"term": {"source_filter": "тп"}},
    ]
    assert build_milvus_exact_filter_expr(nf) == 'actual_filter == "да" AND source_filter == "тп"'


def test_combine_and_cache_key() -> None:
    """Проверяет детерминированный cache-key и объединение Milvus expression."""
    nf = NormalizedExactFilters(by_filter_field={"source_filter": "тп", "actual_filter": "да"})
    assert nf.cache_key_part() == "actual_filter=да|source_filter=тп"
    assert NormalizedExactFilters(by_filter_field={}).cache_key_part() == "no_exact_filters"
    assert combine_milvus_filter_exprs(None, 'source_filter == "тп"') == 'source_filter == "тп"'
    assert combine_milvus_filter_exprs('x', None) == 'x'
    assert combine_milvus_filter_exprs('x', 'y') == 'x AND y'
    assert combine_milvus_filter_exprs(None, None) is None
