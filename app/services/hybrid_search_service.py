from __future__ import annotations

import json
import uuid

from app.api.v1.dto.responses.taskmanager import SearchResult, TaskQueryResponse
from app.common.logger import AISearchLogger
from app.common.storages.interfaces import KeyValueStorageProtocol
from app.domain.exceptions import TooManyRequestsException
from app.infrastructure.adapters.interfaces import (
    IBM25WhooshAdapter,
    ICrossEncoderAdapter,
    ILLMQueue,
    IMilvusDense,
    IOpenSearchAdapter,
    IVLLMAdapter,
)
from app.services.interfaces import IHybridSearchService
from app.settings.config import (
    Settings,
)


class HybridSearchService(IHybridSearchService):
    """Сервис гибридного поиска"""

    def __init__(
        self,
        logger: AISearchLogger,
        redis: KeyValueStorageProtocol,
        settings: Settings,
        os_adapter: IOpenSearchAdapter,
        bm25_adapter: IBM25WhooshAdapter,
        ce_adapter: ICrossEncoderAdapter,
        vllm_client: IVLLMAdapter,
        milvus_dense: IMilvusDense,
        queue: ILLMQueue,
    ):
        self.log = logger
        self.redis = redis
        self.hs = settings.hybrid
        self.sw = settings.switches
        self.llm_queue_settings = settings.llm_queue
        self.sem_settings = settings.llm_global_sem
        self.os = os_adapter
        self.bm25 = bm25_adapter
        self.ce = ce_adapter
        self.vllm = vllm_client
        self.milvus = milvus_dense
        self.queue = queue

    async def enqueue_search(self, query: str, top_k: int) -> TaskQueryResponse:
        """Ставит задачу поиска в очередь."""
        ticket_id = str(uuid.uuid4())
        pack_key = f"{self.llm_queue_settings.ticket_hash_prefix}{ticket_id}:pack"
        result_key = f"{self.llm_queue_settings.ticket_hash_prefix}{ticket_id}:result"
        pack = {
            "type": "search",
            "query": query,
            "top_k": top_k,
        }
        await self.redis.set(
            pack_key, json.dumps(pack, ensure_ascii=False), ttl=self.llm_queue_settings.ticket_ttl
        )
        try:
            _, pos = await self.queue.enqueue({"pack_key": pack_key, "result_key": result_key})
        except OverflowError as e:
            raise TooManyRequestsException("LLM queue overflow, try later") from e
        return TaskQueryResponse(
            task_id=ticket_id,
            url=f"/taskmanager/ticket/{ticket_id}",
            status="queued",
            extra={"position": pos},
        )

    async def enqueue_generate(
        self, query: str, top_k: int, system_prompt: str | None = None
    ) -> TaskQueryResponse:
        """Ставит задачу генерации в очередь."""
        ticket_id = str(uuid.uuid4())
        pack_key = f"{self.llm_queue_settings.ticket_hash_prefix}{ticket_id}:pack"
        result_key = f"{self.llm_queue_settings.ticket_hash_prefix}{ticket_id}:result"
        pack = {
            "type": "generate",
            "query": query,
            "top_k": top_k,
            "system_prompt": system_prompt
            or "Отвечай кратко и по существу, опираясь на предоставленный контекст.",
        }
        await self.redis.set(
            pack_key, json.dumps(pack, ensure_ascii=False), ttl=self.llm_queue_settings.ticket_ttl
        )
        try:
            _, pos = await self.queue.enqueue({"pack_key": pack_key, "result_key": result_key})
        except OverflowError as e:
            raise TooManyRequestsException("LLM queue overflow, try later") from e
        return TaskQueryResponse(
            task_id=ticket_id,
            url=f"/taskmanager/ticket/{ticket_id}",
            status="queued",
            extra={"position": pos},
        )

    async def get_task_status(self, ticket_id: str) -> TaskQueryResponse:
        """Проверяет статус задачи."""
        status = await self.queue.status(ticket_id)
        result_key = f"{self.llm_queue_settings.ticket_hash_prefix}{ticket_id}:result"
        raw = await self.redis.get(result_key)
        results = None
        answer = None
        if raw:
            result_data = json.loads(raw)
            if "results" in result_data:
                results = [SearchResult(**r) for r in result_data["results"]]
            if "answer" in result_data:
                answer = result_data["answer"]
        return TaskQueryResponse(
            task_id=ticket_id,
            url=f"/taskmanager/ticket/{ticket_id}",
            status=status.get("state"),
            extra={"position": status.get("approx_position")},
            results=results,
            answer=answer,
        )
