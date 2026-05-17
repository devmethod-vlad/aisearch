from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.infrastructure.utils.exact_filters import (
    ExactFilterConfig,
    normalize_request_exact_filters,
)
from app.infrastructure.utils.token_filters import (
    MultiValueTokenConfig,
    normalize_request_token_filters,
)
from app.services.hybrid_search_orchestrator import HybridSearchOrchestrator


def _os_config(**overrides: object) -> SimpleNamespace:
    """Возвращает конфиг OpenSearch для unit-стабов с полным набором полей."""
    base = {
        "search_fields": ["question"],
        "output_fields": ["ext_id"],
        "operator": "or",
        "min_should_match": "1",
        "multi_match_type": "best_fields",
        "fuzziness": 0,
        "phrase_field_boosts": {},
        "phrase_slop": 0,
        "bool_min_should_match": 1,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.asyncio
async def test_os_candidates_builds_term_filters() -> None:
    """Проверяет, что lexical-ветка OpenSearch включает token/exact filters в bool.filter."""
    orchestrator = HybridSearchOrchestrator.__new__(HybridSearchOrchestrator)
    os_adapter = SimpleNamespace()
    os_adapter.config = _os_config()
    os_adapter.search = AsyncMock(return_value=[])
    orchestrator.os_adapter = os_adapter

    filters = normalize_request_token_filters(
        {"role": ["Врач"], "component": ["Назначения"]},
        config=MultiValueTokenConfig(
            raw_fields=("role", "product", "component"), token_suffix="_tokens", raw_separator=";"
        ),
    )
    exact_filters = normalize_request_exact_filters(
        {},
        config=ExactFilterConfig(raw_fields=(), field_suffix="_filter"),
    )

    await orchestrator._os_candidates("тест", 5, filters, exact_filters)

    body = os_adapter.search.await_args.args[0]
    assert body["query"]["bool"]["filter"] == [
        {
            "bool": {
                "should": [{"term": {"component_tokens": "назначения"}}],
                "minimum_should_match": 1,
            }
        },
        {
            "bool": {
                "should": [{"term": {"role_tokens": "врач"}}],
                "minimum_should_match": 1,
            }
        }
    ]


@pytest.mark.asyncio
async def test_presearch_ignores_token_filters_in_query_body() -> None:
    """Проверяет, что presearch-запрос не наследует token-фильтры из основного поиска."""
    orchestrator = HybridSearchOrchestrator.__new__(HybridSearchOrchestrator)
    os_adapter = SimpleNamespace()
    os_adapter.config = _os_config(output_fields=["ext_id"])
    os_adapter.search = AsyncMock(return_value=[])
    orchestrator.os_adapter = os_adapter

    result = await orchestrator._presearch_exact_match(
        query="Q1",
        field_name="ext_id",
        use_ce=False,
    )

    assert result is None
    first_body = os_adapter.search.await_args_list[0].args[0]
    assert "filter" not in first_body["query"]["bool"]


@pytest.mark.asyncio
async def test_os_candidates_adds_min_should_match_only_for_or_operator() -> None:
    """Проверяет, что minimum_should_match добавляется в multi_match только для operator=or."""
    orchestrator = HybridSearchOrchestrator.__new__(HybridSearchOrchestrator)
    os_adapter = SimpleNamespace()
    os_adapter.config = SimpleNamespace(
        search_fields=["question", "analysis"],
        operator="or",
        min_should_match="70%",
        multi_match_type="best_fields",
        fuzziness=0,
        output_fields=["ext_id"],
    )
    os_adapter.search = AsyncMock(return_value=[])
    orchestrator.os_adapter = os_adapter

    filters = normalize_request_token_filters(
        {},
        config=MultiValueTokenConfig(
            raw_fields=("role", "product", "component"), token_suffix="_tokens", raw_separator=";"
        ),
    )
    exact_filters = normalize_request_exact_filters(
        {},
        config=ExactFilterConfig(raw_fields=(), field_suffix="_filter"),
    )

    await orchestrator._os_candidates("тест", 5, filters, exact_filters)
    multi_match = os_adapter.search.await_args.args[0]["query"]["bool"]["must"]["multi_match"]

    assert multi_match["type"] == "best_fields"
    assert multi_match["operator"] == "or"
    assert multi_match["minimum_should_match"] == "70%"


@pytest.mark.asyncio
async def test_os_candidates_skips_min_should_match_for_and_operator_and_empty_or() -> None:
    """Проверяет, что для operator=and и пустого min_should_match ключ не добавляется."""
    orchestrator = HybridSearchOrchestrator.__new__(HybridSearchOrchestrator)
    os_adapter = SimpleNamespace()
    os_adapter.config = SimpleNamespace(
        search_fields=["question"],
        operator="and",
        min_should_match="70%",
        multi_match_type="cross_fields",
        fuzziness=0,
        output_fields=["ext_id"],
    )
    os_adapter.search = AsyncMock(return_value=[])
    orchestrator.os_adapter = os_adapter

    filters = normalize_request_token_filters(
        {},
        config=MultiValueTokenConfig(
            raw_fields=("role", "product", "component"), token_suffix="_tokens", raw_separator=";"
        ),
    )
    exact_filters = normalize_request_exact_filters(
        {},
        config=ExactFilterConfig(raw_fields=(), field_suffix="_filter"),
    )
    await orchestrator._os_candidates("тест", 5, filters, exact_filters)
    multi_match = os_adapter.search.await_args.args[0]["query"]["bool"]["must"]["multi_match"]

    assert multi_match["type"] == "cross_fields"
    assert multi_match["operator"] == "and"
    assert "minimum_should_match" not in multi_match

    os_adapter.config.operator = "or"
    os_adapter.config.min_should_match = ""
    await orchestrator._os_candidates("тест2", 5, filters, exact_filters)
    multi_match_empty = os_adapter.search.await_args.args[0]["query"]["bool"]["must"]["multi_match"]
    assert "minimum_should_match" not in multi_match_empty
