from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.infrastructure.utils.token_filters import (
    MultiValueTokenConfig,
    normalize_request_token_filters,
)
from app.services.hybrid_search_orchestrator import HybridSearchOrchestrator


@pytest.mark.asyncio
async def test_os_candidates_builds_term_filters() -> None:
    orchestrator = HybridSearchOrchestrator.__new__(HybridSearchOrchestrator)
    os_adapter = SimpleNamespace()
    os_adapter.config = SimpleNamespace(
        search_fields=["question"], operator="or", fuzziness=0, output_fields=["ext_id"]
    )
    os_adapter.search = AsyncMock(return_value=[])
    orchestrator.os_adapter = os_adapter

    filters = normalize_request_token_filters(
        {"role": ["Врач"]},
        config=MultiValueTokenConfig(
            raw_fields=("role", "product"), token_suffix="_tokens", raw_separator=";"
        ),
    )
    await orchestrator._os_candidates("тест", 5, filters)

    body = os_adapter.search.await_args.args[0]
    assert body["query"]["bool"]["filter"] == [
        {
            "bool": {
                "should": [{"term": {"role_tokens": "врач"}}],
                "minimum_should_match": 1,
            }
        }
    ]


@pytest.mark.asyncio
async def test_presearch_ignores_token_filters_in_query_body() -> None:
    orchestrator = HybridSearchOrchestrator.__new__(HybridSearchOrchestrator)
    os_adapter = SimpleNamespace()
    os_adapter.config = SimpleNamespace(output_fields=["ext_id"])
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
