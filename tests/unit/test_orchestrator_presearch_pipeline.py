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
from app.infrastructure.utils.exact_filters import ExactFilterConfig
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


def _build_orchestrator() -> HybridSearchOrchestrator:
    """Создаёт оркестратор с минимальным набором моков для unit-тестов."""
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
        fusion_mode="weighted_score",
        rrf_k=60,
        top_k=3,
    )
    orchestrator.switches = SimpleNamespace(
        use_opensearch=False,
        use_hybrid=False,
        use_reranker=False,
    )
    orchestrator.short = SimpleNamespace(mode=False)
    orchestrator.normalize_query = False
    orchestrator.os_index_name = "kb_index"
    orchestrator.response_metrics_enabled = False
    orchestrator.log_metrics_enabled = False
    orchestrator.token_filter_config = MultiValueTokenConfig(
        raw_fields=("role", "product", "component"), token_suffix="_tokens", raw_separator=";"
    )
    orchestrator.dense_metric = "unit"
    orchestrator.exact_filter_config = ExactFilterConfig(
        raw_fields=("actual",), field_suffix="_filter"
    )
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


def test_extract_array_filters_validation() -> None:
    """Проверяет валидные и невалидные входы для array_filters."""
    orchestrator = _build_orchestrator()

    assert orchestrator._extract_array_filters({"query": "q"}) == {}
    assert orchestrator._extract_array_filters({"array_filters": None}) == {}
    assert orchestrator._extract_array_filters({"array_filters": {}}) == {}
    assert orchestrator._extract_array_filters({"array_filters": {"role": ["Врач"]}}) == {
        "role": ["Врач"]
    }
    assert orchestrator._extract_array_filters({"array_filters": {"role": []}}) == {
        "role": []
    }
    assert orchestrator._extract_array_filters({"array_filters": {"role": None}}) == {
        "role": None
    }

    with pytest.raises(ValueError, match="invalid array_filters: expected object/dict"):
        orchestrator._extract_array_filters({"array_filters": []})
    with pytest.raises(ValueError, match="invalid array_filters: expected object/dict"):
        orchestrator._extract_array_filters({"array_filters": "role=doctor"})
    with pytest.raises(ValueError, match="invalid array_filters: expected object/dict"):
        orchestrator._extract_array_filters({"array_filters": 123})

    with pytest.raises(
        ValueError,
        match=r"invalid array_filters\.role: expected list of values or null",
    ):
        orchestrator._extract_array_filters({"array_filters": {"role": "Врач"}})
    with pytest.raises(
        ValueError,
        match=r"invalid array_filters\.role: expected list of values or null",
    ):
        orchestrator._extract_array_filters({"array_filters": {"role": 123}})


def test_extract_exact_filters_validation() -> None:
    """Проверяет валидные и невалидные входы для exact_filters."""
    orchestrator = _build_orchestrator()

    assert orchestrator._extract_exact_filters({"query": "q"}) == {}
    assert orchestrator._extract_exact_filters({"exact_filters": None}) == {}
    assert orchestrator._extract_exact_filters({"exact_filters": {}}) == {}
    assert orchestrator._extract_exact_filters({"exact_filters": {"source": "kb"}}) == {
        "source": "kb"
    }
    assert orchestrator._extract_exact_filters({"exact_filters": {"source": 123}}) == {
        "source": 123
    }

    with pytest.raises(ValueError, match="invalid exact_filters: expected object/dict"):
        orchestrator._extract_exact_filters({"exact_filters": []})
    with pytest.raises(ValueError, match="invalid exact_filters: expected object/dict"):
        orchestrator._extract_exact_filters({"exact_filters": "source=kb"})
    with pytest.raises(ValueError, match="invalid exact_filters: expected object/dict"):
        orchestrator._extract_exact_filters({"exact_filters": 123})


@pytest.mark.asyncio
async def test_documents_search_cache_key_depends_on_filters() -> None:
    orchestrator = _build_orchestrator()
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
                    "presearch": {"field": "ext_id"},
                    "array_filters": {
                        "role": ["Врач"],
                        "product": ["ЭМИАС"],
                        "component": ["Назначения"],
                    },
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
                    "presearch": {"field": "ext_id"},
                    "array_filters": {
                        "role": ["Админ"],
                        "product": ["ЭМИАС"],
                        "component": ["Назначения"],
                    },
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
    orchestrator = _build_orchestrator()

    pack_payload = json.dumps(
        {"query": "KB-12345", "top_k": 3, "presearch": {"field": "ext_id"}, "array_filters": {"role": ["Врач"], "product": ["ЭМИАС"], "component": ["Назначения"]}}
    )
    orchestrator.redis.get = AsyncMock(side_effect=[pack_payload, None])
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
    orchestrator = _build_orchestrator()

    pack_payload = json.dumps(
        {"query": "KB-12345", "top_k": 3, "presearch": {"field": "ext_id"}, "array_filters": {"role": ["Врач"], "product": ["ЭМИАС"], "component": ["Назначения"]}}
    )
    orchestrator.redis.get = AsyncMock(side_effect=[pack_payload, None])
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
async def test_documents_search_fails_early_on_invalid_array_filters_type() -> None:
    """Проверяет ранний fail в documents_search при невалидном array_filters."""
    orchestrator = _build_orchestrator()

    pack_payload = json.dumps(
        {"query": "KB-12345", "top_k": 3, "array_filters": []}
    )
    orchestrator.redis.get = AsyncMock(return_value=pack_payload)
    orchestrator.redis.hash_get = AsyncMock(return_value="0")

    result = await orchestrator.documents_search("task-1", "ticket-1", "pack", "result")

    assert result["status"] == "error"
    assert "invalid array_filters" in result["error"]
    orchestrator.queue.set_failed.assert_awaited_once()
    orchestrator.queue.ack.assert_awaited_once()
    orchestrator.vector_db.search.assert_not_called()
    orchestrator.os_adapter.search.assert_not_called()
    orchestrator.redis.set.assert_not_called()


@pytest.mark.asyncio
async def test_documents_search_cache_key_depends_on_data_version() -> None:
    orchestrator = _build_orchestrator()
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
    """Проверяет, что cache hit запрашивается с актуальной data_version."""
    orchestrator = _build_orchestrator()
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


@pytest.mark.asyncio
async def test_documents_search_cache_key_depends_on_exact_filters() -> None:
    """Проверяет, что exact-фильтры участвуют в построении cache key."""
    orchestrator = _build_orchestrator()
    orchestrator_module.get_or_create_search_data_version = AsyncMock(return_value="dv-1")

    pack_key = "pack"

    async def _redis_get(key: str) -> str:
        if key == pack_key:
            return json.dumps({"query": "KB-12345", "top_k": 3, "exact_filters": {"actual": "Да"}})
        return "[]"

    orchestrator.redis.get = AsyncMock(side_effect=_redis_get)
    orchestrator.redis.hash_get = AsyncMock(return_value="0")
    await orchestrator.documents_search("task-1", "ticket-1", pack_key, "result")
    first_cache_key = orchestrator.redis.get.await_args_list[1].args[0]

    async def _redis_get_second(key: str) -> str:
        if key == pack_key:
            return json.dumps({"query": "KB-12345", "top_k": 3, "exact_filters": {"actual": "Нет"}})
        return "[]"

    orchestrator.redis.get = AsyncMock(side_effect=_redis_get_second)
    await orchestrator.documents_search("task-2", "ticket-2", pack_key, "result")
    second_cache_key = orchestrator.redis.get.await_args_list[1].args[0]

    assert first_cache_key != second_cache_key
    assert "actual_filter=" in first_cache_key


@pytest.mark.asyncio
async def test_documents_search_cache_key_depends_on_fusion_mode() -> None:
    """Проверяет, что cache key различается для разных fusion_mode и rrf_k."""
    orchestrator = _build_orchestrator()
    orchestrator_module.get_or_create_search_data_version = AsyncMock(return_value="dv-1")
    pack_payload = json.dumps({"query": "KB-12345", "top_k": 3})
    orchestrator.redis.hash_get = AsyncMock(return_value="0")

    orchestrator.settings.fusion_mode = "weighted_score"
    orchestrator.settings.rrf_k = 60
    orchestrator.redis.get = AsyncMock(side_effect=[pack_payload, None])
    await orchestrator.documents_search("task-1", "ticket-1", "pack", "result")
    weighted_key = orchestrator.redis.get.await_args_list[1].args[0]

    orchestrator.settings.fusion_mode = "rrf"
    orchestrator.settings.rrf_k = 100
    orchestrator.redis.get = AsyncMock(side_effect=[pack_payload, None])
    await orchestrator.documents_search("task-2", "ticket-2", "pack", "result")
    rrf_key = orchestrator.redis.get.await_args_list[1].args[0]

    assert weighted_key != rrf_key
    assert "fusion=weighted_score:rrf_k=60" in weighted_key
    assert "fusion=rrf:rrf_k=100" in rrf_key
