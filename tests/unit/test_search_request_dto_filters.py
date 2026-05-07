from app.api.v1.dto.requests.search import SearchRequest


def test_search_request_accepts_nested_filters_payload() -> None:
    """Проверяет, что DTO принимает новый nested-формат фильтров."""
    payload = {
        "query": "test",
        "top_k": 5,
        "filters": {
            "array_filters": {"role": ["Врач"], "product": ["ЕМИАС"]},
            "exact_filters": {"source": "ТП", "actual": "Да"},
        },
    }

    request = SearchRequest.model_validate(payload)

    assert request.filters is not None
    assert request.filters.array_filters is not None
    assert request.filters.exact_filters is not None


def test_search_request_accepts_payload_without_filters() -> None:
    """Проверяет, что DTO принимает запрос без блока filters."""
    payload = {"query": "test", "top_k": 5}

    request = SearchRequest.model_validate(payload)

    assert request.query == "test"
    assert request.top_k == 5
    assert request.filters is None
