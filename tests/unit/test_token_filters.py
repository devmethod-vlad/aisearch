from app.infrastructure.utils.token_filters import (
    TOKEN_FILTER_CONFIG,
    build_milvus_token_filter_expr,
    build_opensearch_token_filter_clauses,
    normalize_request_token_filters,
    tokenize_multi_value,
)


def test_tokenize_multi_value_normalization() -> None:
    assert tokenize_multi_value("Врач;Медсестра") == ["врач", "медсестра"]
    assert tokenize_multi_value(" Врач ; Медсестра ; ") == ["врач", "медсестра"]
    assert tokenize_multi_value("Врач;врач;МЕДСЕСТРА") == ["врач", "медсестра"]
    assert tokenize_multi_value(None) == []
    assert tokenize_multi_value("") == []


def test_partial_match_is_not_allowed() -> None:
    tokens = tokenize_multi_value("Врач на дому;Медсестра")
    assert tokens == ["врач на дому", "медсестра"]
    assert "врач" not in tokens


def test_config_token_field_mapping() -> None:
    assert TOKEN_FILTER_CONFIG.token_field("role") == "role_tokens"
    assert TOKEN_FILTER_CONFIG.token_field("product") == "product_tokens"


def test_filter_builders_and_cache_part() -> None:
    filters = normalize_request_token_filters({"role": "Врач", "product": "ЭМИАС"})

    assert filters.cache_key_part() == "product_tokens=эмиас|role_tokens=врач"

    os_clauses = build_opensearch_token_filter_clauses(filters)
    assert {"term": {"role_tokens": "врач"}} in os_clauses
    assert {"term": {"product_tokens": "эмиас"}} in os_clauses

    milvus_expr = build_milvus_token_filter_expr(filters)
    assert milvus_expr == (
        'ARRAY_CONTAINS(product_tokens, "эмиас") AND '
        'ARRAY_CONTAINS(role_tokens, "врач")'
    )
