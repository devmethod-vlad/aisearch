from __future__ import annotations

import json
import typing as tp

from celery import shared_task
from dishka import AsyncContainer

from app.api.v1.dto.responses.taskmanager import SearchResult
from app.common.logger import AISearchLogger, LoggerType
from app.common.storages.interfaces import KeyValueStorageProtocol
from app.infrastructure.adapters.interfaces import (
    IBM25WhooshAdapter,
    ICrossEncoderAdapter,
    ILLMQueue,
    IMilvusDense,
    IOpenSearchAdapter,
    IRedisSemaphore,
    IVLLMAdapter,
)
from app.infrastructure.adapters.open_search import OpenSearchAdapter
from app.infrastructure.worker.worker import run_coroutine
from app.settings.config import HybridSearchSettings, SearchSwitches, VLLMSettings, settings


@shared_task(name="search-task", bind=True, max_retries=3, ignore_result=True)
def search_task(
    self: tp.Callable, ticket_id: str, pack_key: str, result_key: str
) -> dict[str, tp.Any]:
    """Выполняет гибридный поиск в фоновом режиме."""

    async def _run():
        container: AsyncContainer = getattr(self, "container", None)

        logger = AISearchLogger(logger_type=LoggerType.CELERY)
        logger.info("Начало выполнения задачи 'search-task'...")

        queue = await container.get(ILLMQueue)
        sem = await container.get(IRedisSemaphore)
        milvus = await container.get(IMilvusDense)
        bm25 = await container.get(IBM25WhooshAdapter)
        ce = await container.get(ICrossEncoderAdapter)
        os_adapter = await container.get(IOpenSearchAdapter)
        settings = await container.get(HybridSearchSettings)
        switches = await container.get(SearchSwitches)
        redis = await container.get(KeyValueStorageProtocol)

        try:
            task_id = getattr(self, "task_id", None) or getattr(self.request, "id", "")
            await queue.set_running(ticket_id, task_id)
        except Exception:
            pass
        raw = await redis.get(pack_key)
        if not raw:
            await queue.set_failed(ticket_id, "missing pack")
            await queue.ack(ticket_id)
            return {"status": "error", "error": "missing pack"}

        pack = json.loads(raw)
        query = pack["query"]
        top_k = int(pack["top_k"])
        try:
            async with sem.acquire():
                cache_key = f"hyb:{query}:{top_k}:{settings.version}"
                cached = await redis.get(cache_key)
                if cached:
                    results = [SearchResult(**x) for x in json.loads(cached)]
                else:
                    dense = await milvus.search(query, top_k=settings.dense_top_k)
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
                    results = _score_and_slice(merged, top_k, use_ce=switches.use_reranker)
                    results = [SearchResult(**r) for r in results]
                    await redis.set(
                        cache_key,
                        json.dumps([r.model_dump() for r in results]),
                        ttl=settings.cache_ttl,
                    )

                payload = {"results": [r.model_dump() for r in results]}
                await redis.set(
                    result_key, json.dumps(payload, ensure_ascii=False), ttl=queue.ticket_ttl
                )
                await queue.set_done(ticket_id)
                return {"status": "ok"}
        except Exception as e:
            await queue.set_failed(ticket_id, str(e))
            raise
        finally:
            try:
                await queue.ack(ticket_id)
            except Exception:
                pass

    return run_coroutine(_run())


@shared_task(name="generate-answer-vllm", bind=True, max_retries=3, ignore_result=True)
def generate_answer_task(
    self: tp.Callable, ticket_id: str, pack_key: str, result_key: str
) -> dict[str, tp.Any]:
    """Выполняет генерацию ответа в фоновом режиме."""

    async def _run() -> None:
        container: AsyncContainer = getattr(self, "container", None)
        logger = AISearchLogger(logger_type=LoggerType.CELERY)
        logger.info("Начало выполнения задачи 'generate-answer-vllm'...")

        queue = await container.get(ILLMQueue)
        sem = await container.get(IRedisSemaphore)
        milvus = await container.get(IMilvusDense)
        bm25 = await container.get(IBM25WhooshAdapter)
        ce = await container.get(ICrossEncoderAdapter)
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
        query = pack["query"]
        top_k = int(pack["top_k"])
        system_prompt = pack["system_prompt"]
        try:
            async with sem.acquire():
                cache_key = f"hyb:{query}:{top_k}:{settings.version}"
                cached = await redis.get(cache_key)
                if cached:
                    results = [SearchResult(**x) for x in json.loads(cached)]
                else:
                    dense = await milvus.search(query, top_k=settings.dense_top_k)
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
                    results = _score_and_slice(merged, top_k, use_ce=switches.use_reranker)
                    results = [SearchResult(**r) for r in results]
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
            try:
                await queue.ack(ticket_id)
            except Exception:
                pass

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
                "operator": os_adapter.s.operator,
                "fuzziness": os_adapter.s.fuzziness,
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


async def _bm25_candidates(bm25: IBM25WhooshAdapter, query: str, k: int) -> list[dict[str, tp.Any]]:
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
            settings.hybrid.w_dense * nd
            + settings.hybrid.w_lex * nl
            + (settings.hybrid.w_ce * nc if use_ce else 0.0)
        )
    items.sort(key=lambda x: x.get("score_final", 0.0), reverse=True)
    return items[:top_k]


def _concat_text(m: dict[str, tp.Any]) -> str:
    return "\n".join([m.get("question", ""), m.get("analysis", ""), m.get("answer", "")])
