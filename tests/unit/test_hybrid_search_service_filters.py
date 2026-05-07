import json
from unittest.mock import AsyncMock

import pytest

from app.common.logger import AISearchLogger, LoggerType
from app.services.hybrid_search_service import HybridSearchService
from app.settings.config import LLMQueueSettings, Settings


@pytest.mark.asyncio
async def test_enqueue_search_pack_contains_filters() -> None:
    """Проверяет, что pack содержит array- и exact-фильтры отдельными словарями."""
    settings = Settings.model_construct(
        llm_queue=LLMQueueSettings.model_construct(
            ticket_hash_prefix="llm:ticket:",
            ticket_ttl=100,
            max_size=10,
            queue_list_key="queue",
            drain_interval_sec=1,
        )
    )
    redis = AsyncMock()
    queue = AsyncMock()
    queue.enqueue.return_value = ("ok", 1)

    service = HybridSearchService(
        logger=AISearchLogger(logger_type=LoggerType.TEST),
        redis=redis,
        settings=settings,
        queue=queue,
    )

    await service.enqueue_search(
        query="q",
        top_k=3,
        array_filters={"role": ["Врач"], "product": ["ЭМИАС"], "component": ["Назначения"]},
        exact_filters={"source": "ТП", "actual": "Да", "second_line": None},
    )

    pack_raw = redis.set.await_args_list[0].args[1]
    pack = json.loads(pack_raw)
    assert pack["array_filters"] == {
        "role": ["Врач"],
        "product": ["ЭМИАС"],
        "component": ["Назначения"],
    }
    assert pack["exact_filters"] == {"source": "ТП", "actual": "Да", "second_line": None}
    assert "role" not in pack
    assert "product" not in pack
    assert "component" not in pack
