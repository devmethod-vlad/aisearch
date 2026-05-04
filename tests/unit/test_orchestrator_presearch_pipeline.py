import json
import uuid
from types import SimpleNamespace, TracebackType
from unittest.mock import AsyncMock

import pytest

import app.services.hybrid_search_orchestrator as orchestrator_module
from app.infrastructure.utils.token_filters import (
    MultiValueTokenConfig,
    build_milvus_token_filter_expr,
    normalize_request_token_filters,
)
from app.services.hybrid_search_orchestrator import HybridSearchOrchestrator


class _AsyncCtx:
    async def __aenter__(self) -> "_AsyncCtx":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return False


class _Sem:
    def acquire(self) -> _AsyncCtx:
        return _AsyncCtx()


class _Uow(_AsyncCtx):
    def __init__(self):
        self.search_request = SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(id=uuid.uuid4()))
        )
        self.commit = AsyncMock()


def _build_orchestrator(*, use_cache: bool) -> HybridSearchOrchestrator:
    orchestrator = HybridSearchOrchestrator.__new__(HybridSearchOrchestrator)
    orchestrator.queue = SimpleNamespace(
        tprefix="ticket:",
        ticket_ttl=60,
        set_running=AsyncMock(),
        set_failed=AsyncMock(),
        set_done=AsyncMock(),
        ack=AsyncMock(),
    )
    orchestrator.sem = _Sem()
    orchestrator.uow = _Uow()
    orchestrator.redis = AsyncMock()
    orchestrator.logger = SimpleNamespace(
        info=lambda *_: None, debug=lambda *_: None, warning=lambda *_: None
    )
    orchestrator._metrics_logger = lambda *_args, **_kwargs: 0.0
    orchestrator.model = SimpleNamespace(encode=lambda *_args, **_kwargs: [[0.1]])
    orchestrator.model_name = "test-model"
    orchestrator.ce_model_name = "test-ce"
    orchestrator.ce = SimpleNamespace(
        rank_fast=lambda _pairs: [0.0], ce_postprocess=lambda scores: scores
    )

    orchestrator.settings = SimpleNamespace(
        presearch_enabled=True,
        presearch_field="ext_id",
        version="v1",
        cache_ttl=120,
        collection_name="docs",
        dense_top_k=5,
        lex_top_k=5,
        merge_by_field="ext_id",
        merge_fields=["question"],
        dense_abs_min=0.0,
        dense_rel_min=0.0,
        lex_rel_min=0.0,
        precut_min_keep=1,
        w_dense=1.0,
        w_lex=0.0,
        w_ce=0.0,
        top_k=3,
    )
    orchestrator.switches = SimpleNamespace(
        use_opensearch=False,
        use_hybrid=False,
        use_reranker=False,
    )
    orchestrator.short = SimpleNamespace(mode=False)
    orchestrator.normalize_query = False
    orchestrator.use_cache = use_cache
    orchestrator.os_index_name = "kb_index"
    orchestrator.enabled_intermediate_results = False
    orchestrator.response_metrics_enabled = False
    orchestrator.log_metrics_enabled = False
    orchestrator.token_filter_config = MultiValueTokenConfig(
        raw_fields=("role", "product", "component"), token_suffix="_tokens", raw_separator=";"
    )
    orchestrator.dense_metric = "unit"
    orchestrator.reranker_pairs_fields = ["question"]
    orchestrator.morph = None

    orchestrator.os_adapter = SimpleNamespace(
        os_schema=SimpleNamespace(index_name="kb_index"),
        config=SimpleNamespace(
            output_fields=["ext_id", "question"],
            search_fields=["question"],
            operator="or",
            fuzziness=0,
        ),
        search=AsyncMock(return_value=[]),
    )
    orchestrator.vector_db = SimpleNamespace(search=AsyncMock(return_value=[]))

    return orchestrator


@pytest.mark.asyncio
async def test_documents_search_cache_key_depends_on_filters() -> None:
    orchestrator = _build_orchestrator(use_cache=True)
    orchestrator_module.get_or_create_search_data_version = AsyncMock(
        return_value="dv-1"
    )

    pack_key = "pack"
    result_key = "result"

    async def _redis_get(key: str) -> str:
        if key == pack_key:
            return json.dumps(
                {
                    "query": "KB-12345",
                    "top_k": 3,
                    "role": ["Врач"],
                    "product": ["ЭМИАС"],
                    "component": ["Назначения"],
                }
            )
        return "[]"

    orchestrator.redis.get = AsyncMock(side_effect=_redis_get)
    orchestrator.redis.hash_get = AsyncMock(return_value="0")

    await orchestrator.documents_search("task-1", "ticket-1", pack_key, result_key)
    first_cache_key = orchestrator.redis.get.await_args_list[1].args[0]

    async def _redis_get_second(key: str) -> str:
        if key == pack_key:
            return json.dumps(
                {
                    "query": "KB-12345",
                    "top_k": 3,
                    "role": ["Админ"],
                    "product": ["ЭМИАС"],
                    "component": ["Назначения"],
                }
            )
        return "[]"

    orchestrator.redis.get = AsyncMock(side_effect=_redis_get_second)
    await orchestrator.documents_search("task-2", "ticket-2", pack_key, result_key)
    second_cache_key = orchestrator.redis.get.await_args_list[1].args[0]

    assert first_cache_key != second_cache_key
    assert ":1:ext_id:" in first_cache_key
    assert "role_tokens=" in first_cache_key
    assert "component_tokens=" in first_cache_key


@pytest.mark.asyncio
async def test_documents_search_keeps_milvus_filter_expr() -> None:
    orchestrator = _build_orchestrator(use_cache=False)

    orchestrator.redis.get = AsyncMock(
        return_value=json.dumps(
            {"query": "KB-12345", "top_k": 3, "role": ["Врач"], "product": ["ЭМИАС"], "component": ["Назначения"]}
        )
    )
    orchestrator.redis.hash_get = AsyncMock(return_value="0")
    orchestrator.vector_db.search = AsyncMock(
        return_value=[{"ext_id": "doc-1", "question": "q", "score_dense": 0.7}]
    )

    await orchestrator.documents_search("task-1", "ticket-1", "pack", "result")

    expected_filters = normalize_request_token_filters(
        {"role": ["Врач"], "product": ["ЭМИАС"], "component": ["Назначения"]},
        config=orchestrator.token_filter_config,
    )
    expected_expr = build_milvus_token_filter_expr(expected_filters)

    assert (
        orchestrator.vector_db.search.await_args.kwargs["filter_expr"] == expected_expr
    )


@pytest.mark.asyncio
async def test_documents_search_injects_presearch_result_even_with_filters() -> None:
    orchestrator = _build_orchestrator(use_cache=False)

    orchestrator.redis.get = AsyncMock(
        return_value=json.dumps(
            {"query": "KB-12345", "top_k": 3, "role": ["Врач"], "product": ["ЭМИАС"], "component": ["Назначения"]}
        )
    )
    orchestrator.redis.hash_get = AsyncMock(return_value="0")
    orchestrator.settings.w_dense = 0.0
    orchestrator.vector_db.search = AsyncMock(return_value=[])

    orchestrator._presearch_exact_match = AsyncMock(
        return_value={
            "ext_id": "KB-12345",
            "question": "точное знание",
            "score_dense": 0.0,
            "score_lex": 1.0,
            "score_final": 1.0,
            "_source": "presearch",
            "sources": ["presearch"],
        }
    )

    await orchestrator.documents_search("task-1", "ticket-1", "pack", "result")

    payload_raw = orchestrator.redis.set.await_args_list[-1].args[1]
    payload = json.loads(payload_raw)

    assert payload["results"][0]["ext_id"] == "KB-12345"
    assert payload["results"][0]["_source"] == "presearch"


@pytest.mark.asyncio
async def test_documents_search_cache_key_depends_on_data_version() -> None:
    orchestrator = _build_orchestrator(use_cache=True)
    pack_payload = json.dumps({"query": "KB-12345", "top_k": 3})
    orchestrator.redis.get = AsyncMock(side_effect=[pack_payload, None, pack_payload, None])
    orchestrator.redis.hash_get = AsyncMock(return_value="0")
    orchestrator_module.get_or_create_search_data_version = AsyncMock(
        side_effect=["dv-1", "dv-2"]
    )

    await orchestrator.documents_search("task-1", "ticket-1", "pack", "result")
    key_v1 = orchestrator.redis.get.await_args_list[1].args[0]
    await orchestrator.documents_search("task-2", "ticket-2", "pack", "result")
    key_v2 = orchestrator.redis.get.await_args_list[3].args[0]

    assert key_v1 != key_v2
    assert ":dv-1:" in key_v1
    assert ":dv-2:" in key_v2


@pytest.mark.asyncio
async def test_documents_search_cache_hit_uses_current_data_version() -> None:
    orchestrator = _build_orchestrator(use_cache=True)
    cached_payload = json.dumps([{"ext_id": "cached"}], ensure_ascii=False)
    orchestrator.redis.get = AsyncMock(
        side_effect=[json.dumps({"query": "KB-1", "top_k": 1}), cached_payload]
    )
    orchestrator.redis.hash_get = AsyncMock(return_value="0")
    orchestrator_module.get_or_create_search_data_version = AsyncMock(
        return_value="dv-current"
    )

    await orchestrator.documents_search("task-1", "ticket-1", "pack", "result")
    cache_key = orchestrator.redis.get.await_args_list[1].args[0]

    assert ":dv-current:" in cache_key
