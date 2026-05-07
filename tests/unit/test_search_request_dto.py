import pytest
from pydantic import ValidationError

from app.api.v1.dto.requests.search import SearchRequest


def test_search_request_defaults_for_runtime_options() -> None:
    """Проверяет значения runtime-параметров по умолчанию."""
    request = SearchRequest.model_validate({"query": "test"})
    assert request.search_use_cache is True
    assert request.show_intermediate_results is False
    assert request.presearch is None


def test_search_request_presearch_field_trimmed() -> None:
    """Проверяет trim presearch.field при валидации DTO."""
    request = SearchRequest.model_validate({"query": "x", "presearch": {"field": " ext_id "}})
    assert request.presearch is not None
    assert request.presearch.field == "ext_id"


def test_search_request_presearch_field_empty_validation_error() -> None:
    """Проверяет ошибку валидации для пустого presearch.field."""
    with pytest.raises(ValidationError):
        SearchRequest.model_validate({"query": "x", "presearch": {"field": "   "}})
