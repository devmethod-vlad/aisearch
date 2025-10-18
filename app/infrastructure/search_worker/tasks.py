from __future__ import annotations

import contextlib
import json
import time
import typing as tp

from celery import shared_task
from dishka import AsyncContainer
import redis.asyncio as redis


from app.api.v1.dto.responses.hybrid_search import SearchResult
from app.common.logger import AISearchLogger, LoggerType
from app.common.storages.interfaces import KeyValueStorageProtocol
from app.infrastructure.adapters.interfaces import (
    IBM25Adapter,
    ICrossEncoderAdapter,
    IOpenSearchAdapter,
    IRedisSemaphore,
    IVLLMAdapter,
)
from app.infrastructure.adapters.light_interfaces import ILLMQueue
from app.infrastructure.adapters.open_search import OpenSearchAdapter
from app.infrastructure.storages.interfaces import IVectorDatabase
from app.infrastructure.utils.nlp import hash_query, normalize_query
from app.infrastructure.search_worker.worker import run_coroutine
from app.services.interfaces import IHybridSearchOrchestrator
from app.settings.config import (
    HybridSearchSettings,
    SearchSwitches,
    VLLMSettings,
    settings as _settings, settings,
)



@shared_task(name="search_task", bind=True, max_retries=3)
def search_task(
    self: tp.Callable, ticket_id: str, pack_key: str, result_key: str
) -> dict[str, tp.Any]:
    """Выполняет гибридный поиск в фоновом режиме."""

    async def _run() -> dict[str, tp.Any]:
        redis_watcher = redis.from_url(str(settings.redis.dsn))
        tprefix = settings.llm_queue.ticket_hash_prefix
        pkey =  settings.llm_queue.processing_list_key or f"{settings.llm_queue.queue_list_key}:processing"

        try:
            container: AsyncContainer | None = getattr(self, "_container", None)
            if not container:
                raise self.retry(exc=Exception("Dishka container not initialized"), countdown=5)

            logger = AISearchLogger(logger_type=LoggerType.CELERY)
            logger.info("Начало выполнения задачи 'search_task'...")

            orchestrator = await container.get(IHybridSearchOrchestrator)
            task_id = getattr(self, "task_id", None) or getattr(self.request, "id", "")
            result = await orchestrator.documents_search(
                task_id=task_id,
                ticket_id=ticket_id,
                pack_key=pack_key,
                result_key=result_key,
            )

            logger.info("Задача 'search_task' выполнена")
            return result
        except Exception as error:
            await redis_watcher.hset(
                f"{tprefix}{ticket_id}",
                mapping={"state": "failed", "error": error, "updated_at": int(time.time())},
            )
            await redis_watcher.lrem(pkey, 1, ticket_id)
            return {"status": "failed"}

    return run_coroutine(_run())


# TODO: Вынести в оркестратор
@shared_task(name="generate-answer-vllm", bind=True, max_retries=3)
def generate_answer_task(
    self: tp.Callable, ticket_id: str, pack_key: str, result_key: str
) -> dict[str, tp.Any]:
    """Выполняет генерацию ответа в фоновом режиме."""

    async def _run() -> None:
        container: AsyncContainer | None = getattr(self, "container", None)
        logger = AISearchLogger(logger_type=LoggerType.CELERY)
        logger.info("Начало выполнения задачи 'generate-answer-vllm'...")

        queue = await container.get(ILLMQueue)
        sem = await container.get(IRedisSemaphore)
        bm25 = await container.get(IBM25Adapter)
        ce = await container.get(ICrossEncoderAdapter)
        milvus = await container.get(IVectorDatabase)
        os_adapter = await container.get(IOpenSearchAdapter)
        settings = await container.get(HybridSearchSettings)
        switches = await container.get(SearchSwitches)
        redis = await container.get(KeyValueStorageProtocol)
        vllm_settings = await container.get(VLLMSettings)
        vllm = await container.get(IVLLMAdapter)

        await queue.set_running(ticket_id, self.request.id)
        raw = await redis.get(pack_key)
        if not raw:
            await queue.set_failed(ticket_id, "missing pack")
            await queue.ack(ticket_id)
            return {"status": "error", "error": "missing pack"}

        pack = json.loads(raw)
        query = normalize_query(pack["query"])
        query_hash = hash_query(query)
        query_vector = self._model.encode([query], convert_to_numpy=True)[0]
        top_k = int(pack["top_k"])
        system_prompt = pack["system_prompt"]
        try:
            async with sem.acquire():
                cache_key = f"hyb:{query_hash}:{top_k}:{settings.version}"
                cached = await redis.get(cache_key)
                if cached:
                    results = [SearchResult(**x) for x in json.loads(cached)]
                else:
                    # dense = await milvus.search(query, top_k=settings.dense_top_k)
                    dense = await milvus.search(
                        collection_name=settings.collection_name,
                        query_vector=query_vector,
                        top_k=top_k,
                    )
                    lex = []
                    if switches.use_hybrid:
                        if switches.use_opensearch:
                            lex = await _os_candidates(os_adapter, query, settings.lex_top_k)
                        elif switches.use_bm25:
                            lex = await _bm25_candidates(bm25, query, settings.lex_top_k)
                    merged = _merge_candidates(dense, lex)
                    if switches.use_reranker and merged:
                        pairs = [(query, _concat_text(m)) for m in merged]
                        ce_scores = ce.rank(pairs)
                        for m, s in zip(merged, ce_scores):
                            m["score_ce"] = float(s)
                    _results = _score_and_slice(merged, top_k, use_ce=switches.use_reranker)
                    results = [SearchResult(**r) for r in _results]
                    await redis.set(
                        cache_key,
                        json.dumps([r.model_dump() for r in results]),
                        ttl=settings.cache_ttl,
                    )

                if results:
                    best = results[0]
                    ctx = f"- {best.answer or ''}"
                else:
                    ctx = "(Контекст отсутствует)"
                user_prompt = f"Вопрос: {query}\n\nКонтекст:\n{ctx}\n\nСформулируй понятный ответ на русском языке."

                answer_text = await vllm.chat_completion(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=vllm_settings.max_output_tokens,
                    temperature=vllm_settings.temperature,
                    top_p=vllm_settings.top_p,
                )
                payload = {"answer": answer_text, "results": [r.model_dump() for r in results]}
                await redis.set(
                    result_key, json.dumps(payload, ensure_ascii=False), ttl=queue.ticket_ttl
                )
                await queue.set_done(ticket_id)
                return {"status": "ok"}
        except Exception as e:
            await queue.set_failed(ticket_id, str(e))
            raise
        finally:
            with contextlib.suppress(Exception):
                await queue.ack(ticket_id)

    return run_coroutine(_run())


# Вспомогательные функции (без изменений)
async def _os_candidates(
    os_adapter: OpenSearchAdapter, query: str, k: int
) -> list[dict[str, tp.Any]]:
    fields = ["question", "analysis", "answer"]
    body = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": fields,
                "operator": os_adapter.config.operator,
                "fuzziness": os_adapter.config.fuzziness,
            }
        }
    }
    hits = os_adapter.search(body, size=k)
    out = []
    for h in hits:
        src = h.get("_source", {})
        out.append(
            {
                "ext_id": src.get("ext_id", h.get("_id")),
                "question": src.get("question", ""),
                "analysis": src.get("analysis", ""),
                "answer": src.get("answer", ""),
                "score_lex": float(h.get("_score", 0.0)),
                "source": "opensearch",
            }
        )
    return out


async def _bm25_candidates(bm25: IBM25Adapter, query: str, k: int) -> list[dict[str, tp.Any]]:
    rows = bm25.search(query, top_k=k)
    out = []
    for r in rows:
        out.append(
            {
                "ext_id": r.get("ext_id"),
                "question": r.get("question", ""),
                "analysis": r.get("analysis", ""),
                "answer": r.get("answer", ""),
                "score_lex": float(r.get("score_bm25", 0.0)),
                "source": "bm25",
            }
        )
    return out


def _merge_candidates(
    dense: list[dict[str, tp.Any]], lex: list[dict[str, tp.Any]]
) -> list[dict[str, tp.Any]]:
    by_id: dict[str, dict[str, tp.Any]] = {}
    for d in dense:
        by_id[d["ext_id"]] = {**d}
    for l in lex:  # noqa: E741
        x = by_id.get(l["ext_id"]) or {}
        x.setdefault("ext_id", l["ext_id"])
        for f in ("question", "analysis", "answer"):
            if not x.get(f):
                x[f] = l.get(f, "")
        prev = x.get("score_lex", 0.0)
        x["score_lex"] = max(prev, float(l.get("score_lex", 0.0)))
        x["source"] = l["source"]
        by_id[x["ext_id"]] = x
    return list(by_id.values())


def _score_and_slice(
    items: list[dict[str, tp.Any]], top_k: int, *, use_ce: bool
) -> list[dict[str, tp.Any]]:
    if not items:
        return []

    def _max(key: str) -> float:
        return max((i.get(key, 0.0) for i in items), default=1.0) or 1.0

    m_dense = _max("score_dense")
    m_lex = _max("score_lex")
    m_ce = _max("score_ce") if use_ce else 1.0
    for it in items:
        nd = (it.get("score_dense", 0.0) / m_dense) if m_dense else 0.0
        nl = (it.get("score_lex", 0.0) / m_lex) if m_lex else 0.0
        nc = (it.get("score_ce", 0.0) / m_ce) if use_ce else 0.0
        it["score_final"] = (
            _settings.hybrid.w_dense * nd
            + _settings.hybrid.w_lex * nl
            + (_settings.hybrid.w_ce * nc if use_ce else 0.0)
        )
    items.sort(key=lambda x: x.get("score_final", 0.0), reverse=True)
    return items[:top_k]


def _concat_text(m: dict[str, tp.Any]) -> str:
    return "\n".join([m.get("question", ""), m.get("analysis", ""), m.get("answer", "")])
