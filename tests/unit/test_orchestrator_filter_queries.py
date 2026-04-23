from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.infrastructure.utils.token_filters import normalize_request_token_filters
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

    filters = normalize_request_token_filters({"role": "Врач"})
    await orchestrator._os_candidates("тест", 5, filters)

    body = os_adapter.search.await_args.args[0]
    assert {"term": {"role_tokens": "врач"}} in body["query"]["bool"]["filter"]


@pytest.mark.asyncio
async def test_presearch_uses_same_filter_clauses() -> None:
    orchestrator = HybridSearchOrchestrator.__new__(HybridSearchOrchestrator)
    os_adapter = SimpleNamespace()
    os_adapter.config = SimpleNamespace(output_fields=["ext_id"])
    os_adapter.search = AsyncMock(return_value=[])
    orchestrator.os_adapter = os_adapter

    filters = normalize_request_token_filters({"product": "ЭМИАС"})
    result = await orchestrator._presearch_exact_match(
        query="Q1",
        field_name="ext_id",
        use_ce=False,
        token_filters=filters,
    )

    assert result is None
    first_body = os_adapter.search.await_args_list[0].args[0]
    assert {"term": {"product_tokens": "эмиас"}} in first_body["query"]["bool"]["filter"]
