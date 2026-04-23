from app.infrastructure.utils.token_filters import (
    MultiValueTokenConfig,
    build_milvus_token_filter_expr,
    build_opensearch_token_filter_clauses,
    normalize_request_filter_values,
    normalize_request_token_filters,
    tokenize_record_raw_value,
)


def test_tokenize_multi_value_normalization() -> None:
    assert tokenize_record_raw_value("Врач;Медсестра", separator=";") == [
        "врач",
        "медсестра",
    ]
    assert tokenize_record_raw_value(" Врач ; Медсестра ; ", separator=";") == [
        "врач",
        "медсестра",
    ]
    assert tokenize_record_raw_value("Врач;врач;МЕДСЕСТРА", separator=";") == [
        "врач",
        "медсестра",
    ]
    assert tokenize_record_raw_value(None, separator=";") == []
    assert tokenize_record_raw_value("", separator=";") == []


def test_partial_match_is_not_allowed() -> None:
    tokens = tokenize_record_raw_value("Врач на дому;Медсестра", separator=";")
    assert tokens == ["врач на дому", "медсестра"]
    assert "врач" not in tokens


def test_config_token_field_mapping() -> None:
    config = MultiValueTokenConfig(
        raw_fields=("role", "product"),
        token_suffix="_tokens",
        raw_separator=";",
    )
    assert config.token_field("role") == "role_tokens"
    assert config.token_field("product") == "product_tokens"


def test_request_filter_normalization_from_arrays() -> None:
    assert normalize_request_filter_values(["Врач", "Медсестра"]) == (
        "врач",
        "медсестра",
    )
    assert normalize_request_filter_values([" Врач ", "", "МЕДСЕСТРА", "врач"]) == (
        "врач",
        "медсестра",
    )
    assert normalize_request_filter_values(None) == ()
    assert normalize_request_filter_values(["ЭМИАС;ЛИС"]) == ("эмиас;лис",)


def test_filter_builders_and_cache_part() -> None:
    config = MultiValueTokenConfig(
        raw_fields=("role", "product"),
        token_suffix="_tokens",
        raw_separator=";",
    )
    filters = normalize_request_token_filters(
        {"role": ["Врач"], "product": ["ЭМИАС"]},
        config=config,
    )

    assert filters.cache_key_part() == "product_tokens=эмиас|role_tokens=врач"

    os_clauses = build_opensearch_token_filter_clauses(filters)
    assert os_clauses == [
        {
            "bool": {
                "should": [{"term": {"product_tokens": "эмиас"}}],
                "minimum_should_match": 1,
            }
        },
        {
            "bool": {
                "should": [{"term": {"role_tokens": "врач"}}],
                "minimum_should_match": 1,
            }
        },
    ]

    milvus_expr = build_milvus_token_filter_expr(filters)
    assert milvus_expr == (
        'ARRAY_CONTAINS(product_tokens, "эмиас") AND '
        'ARRAY_CONTAINS(role_tokens, "врач")'
    )


def test_builders_or_inside_and_between_groups() -> None:
    filters = normalize_request_token_filters(
        {
            "role": ["Врач", "Медсестра"],
            "product": ["ЭМИАС", "ЛИС"],
        },
        config=MultiValueTokenConfig(
            raw_fields=("role", "product"),
            token_suffix="_tokens",
            raw_separator=";",
        ),
    )

    os_clauses = build_opensearch_token_filter_clauses(filters)
    assert os_clauses[0]["bool"]["minimum_should_match"] == 1
    assert len(os_clauses[0]["bool"]["should"]) == 2
    assert len(os_clauses[1]["bool"]["should"]) == 2

    milvus_expr = build_milvus_token_filter_expr(filters)
    assert milvus_expr == (
        '(ARRAY_CONTAINS(product_tokens, "эмиас") OR ARRAY_CONTAINS(product_tokens, "лис")) '
        'AND (ARRAY_CONTAINS(role_tokens, "врач") OR ARRAY_CONTAINS(role_tokens, "медсестра"))'
    )


def test_cache_key_is_deterministic_after_normalization() -> None:
    config = MultiValueTokenConfig(
        raw_fields=("product", "role"),
        token_suffix="_tokens",
        raw_separator=";",
    )
    left = normalize_request_token_filters(
        {"role": [" Врач ", "врач"], "product": ["ЛИС", "ЭМИАС"]},
        config=config,
    )
    right = normalize_request_token_filters(
        {"product": ["ЛИС", "ЭМИАС"], "role": ["Врач"]},
        config=config,
    )
    assert left.cache_key_part() == right.cache_key_part()
