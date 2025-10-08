import asyncio
import contextlib
import json
import typing as tp

from sentence_transformers import SentenceTransformer, CrossEncoder

from app.api.v1.dto.responses.hybrid_search import SearchResult
from app.common.storages.interfaces import KeyValueStorageProtocol
from app.infrastructure.adapters.interfaces import (
    IBM25Adapter,
    ICrossEncoderAdapter,
    ILLMQueue,
    IOpenSearchAdapter,
    IRedisSemaphore,
)
from app.infrastructure.storages.interfaces import IVectorDatabase
from app.infrastructure.utils.nlp import hash_query, normalize_query
from app.services.interfaces import IHybridSearchOrchestrator
from app.settings.config import Settings


class HybridSearchOrchestrator(IHybridSearchOrchestrator):
    """Оркестратор гибридного поиска"""

    def __init__(
        self,
        queue: ILLMQueue,
        sem: IRedisSemaphore,
        vector_db: IVectorDatabase,
        bm25: IBM25Adapter,
        ce: ICrossEncoderAdapter,
        os_adapter: IOpenSearchAdapter,
        settings: Settings,
        redis: KeyValueStorageProtocol,
    ):
        self.queue = queue
        self.sem = sem
        self.vector_db = vector_db
        self.bm25 = bm25
        self.ce = ce
        self.os_adapter = os_adapter
        self.redis = redis
        self.settings = settings.hybrid
        self.switches = settings.switches
        self.use_cache = settings.app.use_cache

    # def initialize_orchestrator(self):
    #     print("Инициализация оркестратора")

    async def documents_search(
        self,
        task_id: str,
        ticket_id: str,
        pack_key: str,
        result_key: str,
        model: SentenceTransformer,
        ce_model: CrossEncoder,
    ) -> dict[str, str]:
        """Поиск по документам"""
        import time
        start_total = time.perf_counter()

        await self.queue.set_running(ticket_id, task_id)

        # Получение pack из Redis
        redis_get_start = time.perf_counter()
        raw = await self.redis.get(pack_key)
        redis_get_time = time.perf_counter() - redis_get_start
        print(f"⏱️  Redis GET pack: {redis_get_time:.4f} сек")

        if not raw:
            await self.queue.set_failed(ticket_id, "missing pack")
            await self.queue.ack(ticket_id)
            return {"status": "error", "error": "missing pack"}

        # Парсинг JSON
        json_parse_start = time.perf_counter()
        pack = json.loads(raw)
        json_parse_time = time.perf_counter() - json_parse_start
        print(f"⏱️  JSON parse: {json_parse_time:.4f} сек")
        normalize_start_time = time.perf_counter()
        query = normalize_query(pack["query"])
        query_hash = hash_query(query)
        top_k = int(pack["top_k"])
        need_ack = True
        normalize_end_time = time.perf_counter()
        norm_time = normalize_end_time - normalize_start_time
        print(f"⏱️  NORMALIZE_TIME: {norm_time:.4f} сек")
        try:
            # Ожидание семафора
            sem_acquire_start = time.perf_counter()
            async with self.sem.acquire():
                sem_acquire_time = time.perf_counter() - sem_acquire_start
                print(f"⏱️  Semaphore acquire: {sem_acquire_time:.4f} сек")

                # Векторизация запроса
                encode_start = time.perf_counter()
                query_vector = (await asyncio.to_thread(model.encode, [query], convert_to_numpy=True))[0]
                encode_time = time.perf_counter() - encode_start
                print(f"⏱️  Model encode: {encode_time:.4f} сек")
                cache_key = f"hyb:{query_hash}:{top_k}:{self.settings.version}"
                if self.use_cache:
                    cache_get_start = time.perf_counter()
                    cached = await self.redis.get(cache_key)
                    cache_get_time = time.perf_counter() - cache_get_start
                    print(f"⏱️  Cache GET: {cache_get_time:.4f} сек")
                else:
                    cached = None
                if cached:
                    print("✅ Используется кешированный результат")
                    cache_parse_start = time.perf_counter()
                    results = [SearchResult(**x) for x in json.loads(cached)]
                    cache_parse_time = time.perf_counter() - cache_parse_start
                    print(f"⏱️  Cache parse: {cache_parse_time:.4f} сек")
                else:
                    print("🔍 Выполняется новый поиск")
                    # Векторный поиск
                    vector_search_start = time.perf_counter()
                    dense = await self.vector_db.search(
                        collection_name=self.settings.collection_name,
                        query_vector=query_vector,
                        top_k=top_k,
                    )
                    vector_search_time = time.perf_counter() - vector_search_start
                    print(f"⏱️  Vector search: {vector_search_time:.4f} сек")

                    lex = []
                    lex_search_time = 0

                    # Лексический поиск
                    if self.switches.use_hybrid:
                        lex_search_start = time.perf_counter()
                        if self.switches.use_opensearch:
                            lex = await self._os_candidates(query, self.settings.lex_top_k)
                            print(f"🔍 OpenSearch candidates: {len(lex)} результатов")
                        elif self.switches.use_bm25:
                            lex = await self._bm25_candidates(query, self.settings.lex_top_k)
                            print(f"🔍 BM25 candidates: {len(lex)} результатов")
                        lex_search_time = time.perf_counter() - lex_search_start
                        print(f"⏱️  Lexical search: {lex_search_time:.4f} сек")

                    # Слияние результатов
                    merge_start = time.perf_counter()
                    merged = self._merge_candidates(dense, lex)
                    merge_time = time.perf_counter() - merge_start
                    print(f"⏱️  Merge candidates: {merge_time:.4f} сек")
                    print(f"📊 После слияния: {len(merged)} кандидатов")

                    # Re-ranking
                    rerank_time = 0
                    if self.switches.use_reranker and merged:
                        print("🎯 Применяется re-ranking")
                        rerank_start = time.perf_counter()

                        pairs_prep_start = time.perf_counter()
                        pairs = [(query, self._concat_text(m)) for m in merged]
                        pairs_prep_time = time.perf_counter() - pairs_prep_start
                        print(f"⏱️  Pairs preparation: {pairs_prep_time:.4f} сек")

                        ce_rank_start = time.perf_counter()
                        ce_scores = await asyncio.to_thread(self.ce.rank, ce_model, pairs)
                        ce_rank_time = time.perf_counter() - ce_rank_start
                        print(f"⏱️  Cross-encoder rank: {ce_rank_time:.4f} сек")

                        for m, s in zip(merged, ce_scores):
                            m["score_ce"] = float(s)
                        rerank_time = time.perf_counter() - rerank_start
                        print(f"⏱️  Total re-ranking: {rerank_time:.4f} сек")

                    # Финальная оценка и сортировка
                    scoring_start = time.perf_counter()
                    _results = self._score_and_slice(
                        merged, top_k, use_ce=self.switches.use_reranker
                    )
                    scoring_time = time.perf_counter() - scoring_start
                    print(f"⏱️  Final scoring: {scoring_time:.4f} сек")

                    results = [SearchResult(**r) for r in _results]

                    if self.use_cache:
                        cache_set_start = time.perf_counter()
                        await self.redis.set(
                            cache_key,
                            json.dumps([r.model_dump() for r in results]),
                            ttl=self.settings.cache_ttl,
                        )
                        cache_set_time = time.perf_counter() - cache_set_start
                        print(f"⏱️  Cache SET: {cache_set_time:.4f} сек")

                # Сохранение результатов
                result_prep_start = time.perf_counter()
                payload = {"results": [r.model_dump() for r in results]}
                result_prep_time = time.perf_counter() - result_prep_start
                print(f"⏱️  Result preparation: {result_prep_time:.4f} сек")

                result_set_start = time.perf_counter()
                await self.redis.set(
                    result_key, json.dumps(payload, ensure_ascii=False), ttl=self.queue.ticket_ttl
                )
                result_set_time = time.perf_counter() - result_set_start
                print(f"⏱️  Result SET: {result_set_time:.4f} сек")

                queue_done_start = time.perf_counter()
                await self.queue.set_done(ticket_id)
                queue_done_time = time.perf_counter() - queue_done_start
                print(f"⏱️  Queue set done: {queue_done_time:.4f} сек")

                total_time = time.perf_counter() - start_total
                print(f"✅ ПОИСК ЗАВЕРШЕН: {total_time:.4f} сек")
                print(f"📊 Итоговые результаты: {len(results)} документов")

                return {"status": "ok"}

        except TimeoutError as e:
            timeout_time = time.perf_counter() - start_total
            print(f"⏰ TIMEOUT через {timeout_time:.4f} сек: {e}")
            await self.queue.requeue(ticket_id, reason="global semaphore busy")
            need_ack = False
            return {"status": "requeued"}
        except Exception as e:
            error_time = time.perf_counter() - start_total
            print(f"❌ ОШИБКА через {error_time:.4f} сек: {e}")
            await self.queue.set_failed(ticket_id, str(e))
            raise
        finally:
            if need_ack:
                ack_start = time.perf_counter()
                with contextlib.suppress(Exception):
                    await self.queue.ack(ticket_id)
                ack_time = time.perf_counter() - ack_start
                print(f"⏱️  Queue ACK: {ack_time:.4f} сек")

            final_total = time.perf_counter() - start_total
            print(f"🏁 ОБЩЕЕ ВРЕМЯ ВЫПОЛНЕНИЯ: {final_total:.4f} сек")
            print("=" * 60)

    async def _os_candidates(self, query: str, k: int) -> list[dict[str, tp.Any]]:
        os_adapter = self.os_adapter
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
        hits = await asyncio.to_thread(os_adapter.search, body, k)
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

    async def _bm25_candidates(self, query: str, k: int) -> list[dict[str, tp.Any]]:
        rows = await asyncio.to_thread(self.bm25.search, query, k)
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
        self, dense: list[dict[str, tp.Any]], lex: list[dict[str, tp.Any]]
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
        self, items: list[dict[str, tp.Any]], top_k: int, *, use_ce: bool
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
                self.settings.w_dense * nd
                + self.settings.w_lex * nl
                + (self.settings.w_ce * nc if use_ce else 0.0)
            )
        items.sort(key=lambda x: x.get("score_final", 0.0), reverse=True)
        return items[:top_k]

    def _concat_text(self, m: dict[str, tp.Any]) -> str:
        return "\n".join([m.get("question", ""), m.get("analysis", ""), m.get("answer", "")])
