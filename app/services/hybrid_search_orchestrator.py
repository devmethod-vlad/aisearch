import asyncio
import contextlib
import json
import time
import typing as tp

from sentence_transformers import SentenceTransformer

from app.common.logger import AISearchLogger
from app.common.storages.interfaces import KeyValueStorageProtocol
from app.infrastructure.adapters.interfaces import (
    IBM25Adapter,
    ICrossEncoderAdapter,
    IOpenSearchAdapter,
    IRedisSemaphore,
)
from app.infrastructure.adapters.light_interfaces import ILLMQueue
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
        logger: AISearchLogger,
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
        self.ce_settings = settings.reranker
        self.model_name = settings.milvus.model_name
        self.use_cache = settings.app.use_cache
        self.normalize_query = settings.app.normalize_query
        self.model = SentenceTransformer(settings.milvus.model_name)

        self.log_metrics_enabled = settings.search_metrics.log_metrics_enabled
        self.response_metrics_enabled = settings.search_metrics.response_metrics_enabled
        
        self.dense_metric = settings.milvus.metric_type
        self.reranker_pairs_fields = settings.reranker.pairs_fields
        self.logger = logger

    async def documents_search(
            self,
            task_id: str,
            ticket_id: str,
            pack_key: str,
            result_key: str,
    ) -> dict[str, str]:
        """Поиск по документам с логированием и сбором всех метрик."""
        start_total = time.perf_counter()
        metrics: dict[str, float] = {}

        await self.queue.set_running(ticket_id, task_id)

        # ---- Redis GET pack ----
        redis_get_start = time.perf_counter()
        raw = await self.redis.get(pack_key)
        metrics["redis_get_time"] = self._metrics_logger("🕒 Redis GET pack", redis_get_start)

        if not raw:
            await self.queue.set_failed(ticket_id, "missing pack")
            await self.queue.ack(ticket_id)
            return {"status": "error", "error": "missing pack"}

        # ---- JSON parse ----
        json_parse_start = time.perf_counter()
        pack = json.loads(raw)
        metrics["json_parse_time"] = self._metrics_logger("🕒 JSON parse", json_parse_start)

        # ---- Normalize query ----
        if self.normalize_query:
            normalize_start = time.perf_counter()
            query = normalize_query(pack["query"])
            query_hash = hash_query(query)
            top_k = int(pack["top_k"])
            metrics["normalize_time"] = self._metrics_logger("🕒 Normalize query", normalize_start)
        else:
            query = pack["query"]
            top_k = int(pack["top_k"])
            query_hash = pack["query"]
        need_ack = True
        try:
            async with self.sem.acquire():
                metrics["semaphore_acquire_time"] = self._metrics_logger("🕒 Semaphore acquire", time.perf_counter())

                cache_key = f"hyb:{query_hash}:{top_k}:{self.settings.version}"
                cached = await self.redis.get(cache_key) if self.use_cache else None

                if cached:
                    self.logger.info("📦 Выдаем результат из кеша")
                    cache_parse_start = time.perf_counter()
                    results = [json.loads(cached)]
                    metrics["cache_parse_time"] = self._metrics_logger("🕒 Cache parse", cache_parse_start)
                else:
                    # ---- Embedding ----
                    encode_start = time.perf_counter()
                    query_vector = (await asyncio.to_thread(
                        self.model.encode, [query], convert_to_numpy=True, normalize_embeddings=True
                    ))[0]
                    metrics["embedding_time"] = self._metrics_logger("🕒 Model encode", encode_start)

                    # ---- Dense & Lex search ----
                    async def _dense_task():
                        start = time.perf_counter()
                        res = await self.vector_db.search(
                            collection_name=self.settings.collection_name,
                            query_vector=query_vector,
                            top_k=self.settings.dense_top_k,
                        )
                        metrics["vector_search_time"] = self._metrics_logger("🕒 Milvus search", start)
                        return res

                    async def _lex_task():
                        start = time.perf_counter()
                        if self.switches.use_hybrid:
                            if self.switches.use_opensearch:
                                res = await self._os_candidates(query, self.settings.lex_top_k)
                            elif self.switches.use_bm25:
                                res = await self._bm25_candidates(query, self.settings.lex_top_k)
                            else:
                                res = []
                        else:
                            res = []
                        metrics["lexical_search_time"] = self._metrics_logger("🕒 Lexical search", start)
                        return res


                    dense, lex = await asyncio.gather(_dense_task(), _lex_task())

                    print("dense results: ", dense)
                    print("len(dense) : ", len(dense))
                    print("lex results: ", lex)
                    print("len(lex): ", len(lex))

                    dense = self._precut_dense(dense)
                    lex = self._precut_lex(lex)

                    print("dense after precut results: ", dense)
                    print("len(dense) : ", len(dense))
                    print("lex after precut  results: ", lex)
                    print("len(lex): ", len(lex))

                    for d in dense:
                        d["score_dense"] = self._dense_to_unit(d.get("score_dense", 0.0))

                    print("dense after norm results: ", dense)

                    merged = self._merge_candidates(dense, lex)

                    print("merged: ", merged)
                    print("len(merged) : ", len(merged))

                    # ---- Cross-encoder ----
                    if self.switches.use_reranker and merged:
                        start = time.perf_counter()
                        pairs = [(query, self._concat_text(m)) for m in merged]
                        print("pairs: ", pairs)
                        # scores = await asyncio.to_thread(self.ce.rank, pairs)
                        scores = await asyncio.to_thread(self.ce.rank_fast, pairs)
                        scores = self.ce.ce_postprocess(scores)
                        metrics["cross_encoder_time"] = self._metrics_logger("🕒 Cross-encoder rank", start)
                        for m, s in zip(merged, scores):
                            m["score_ce"] = float(s)
                        print("merged after rerank: ", merged)

                    results = self._score_and_slice(merged, top_k, use_ce=self.switches.use_reranker)

                    print("results (_score_and_slice): ", results)
                    print("len(results) : ", len(results))

                    if self.use_cache:
                        await self.redis.set(
                            cache_key,
                            results,
                            ttl=self.settings.cache_ttl,
                        )

                # ---- Total time ----
                metrics["total_search_time"] = self._metrics_logger("🕒 Total search", start_total)

                # ---- Payload ----
                payload = {
                    "results": results
                }
                if self.response_metrics_enabled:
                    payload["metrics"] = {
                        "embedding_time": metrics.get("embedding_time"),
                        "vector_search_time": metrics.get("vector_search_time"),
                        "opensearch_time": metrics.get("lexical_search_time") if self.switches.use_opensearch else None,
                        "bm25_time": metrics.get("lexical_search_time") if self.switches.use_bm25 else None,
                        "cross_encoder_time": metrics.get("cross_encoder_time"),
                        "total_time": metrics.get("total_search_time"),
                        "reranker_enabled": self.switches.use_reranker,
                        "open_search_enabled": self.switches.use_opensearch,
                        "bm_25_enabled": self.switches.use_bm25,
                        "hybrid_dense_top_k": self.settings.dense_top_k,
                        "hybrid_lex_top_k": self.settings.lex_top_k,
                        "hybrid_top_k": self.settings.top_k,
                        "hybrid_w_ce": self.settings.w_ce,
                        "hybrid_w_dense": self.settings.w_dense,
                        "hybrid_w_lex": self.settings.w_lex,
                        "encoder_model": self.model_name,
                        "reranker_model": self.ce_settings.model_name,

                    }

                await self.redis.set(result_key, json.dumps(payload, ensure_ascii=False), ttl=self.queue.ticket_ttl)
                await self.queue.set_done(ticket_id)

                return {"status": "ok"}

        finally:
            if need_ack:
                ack_start = time.perf_counter()
                with contextlib.suppress(Exception):
                    await self.queue.ack(ticket_id)
                metrics["queue_ack_time"] = self._metrics_logger("🕒 Queue ACK", ack_start)

    async def _os_candidates(self, query: str, k: int) -> list[dict[str, tp.Any]]:
        os_adapter = self.os_adapter
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": os_adapter.config.search_fields,
                    "operator": os_adapter.config.operator,
                    "fuzziness": os_adapter.config.fuzziness,
                }
            }
        }

        hits = await asyncio.to_thread(os_adapter.search, body, k)

        out: list[dict[str, tp.Any]] = []
        output_fields = os_adapter.config.output_fields
        
        for h in hits:
            src = h.get("_source", {})
            raw = float(h.get("_score", 0.0))  # сырое значение от OpenSearch
            
            result_item = {
                "score_lex_raw": raw,  # сохраняем «как есть»
                "score_lex": raw,      # пока сырое — нормализуем ниже
                "_source": "opensearch",
            }
            
            for field in output_fields:
                result_item[field] = src.get(field, "")
            
            # if "ext_id" in output_fields and not result_item["ext_id"]:
            #     result_item["ext_id"] = h.get("_id", "")
                
            out.append(result_item)

        # нормализация в рамках одного запроса: s = raw / top
        if out:
            top = max(x["score_lex"] for x in out) or 0.0
            if top > 0.0:
                for x in out:
                    x["score_lex"] = x["score_lex"] / top
            else:
                # если все нули (редко, но бывает) — не даём NaN/inf
                for x in out:
                    x["score_lex"] = 0.0

        return out

    async def _bm25_candidates(self, query: str, k: int) -> list[dict[str, tp.Any]]:
        rows = await asyncio.to_thread(self.bm25.search, query, k)

        out: list[dict[str, tp.Any]] = []
        output_fields = self.bm25.config.output_fields
        
        for r in rows:
            raw = float(r.get("score_bm25", 0.0))
            
            result_item = {
                "score_lex_raw": raw,  # сырое значение
                "score_lex": raw,      # нормализуем ниже
                "_source": "bm25",
            }
            
            for field in output_fields:
                result_item[field] = r.get(field, "")
                
            out.append(result_item)

        if out:
            top = max(x["score_lex"] for x in out) or 0.0
            if top > 0.0:
                for x in out:
                    x["score_lex"] = x["score_lex"] / top
            else:
                for x in out:
                    x["score_lex"] = 0.0

        return out

    def _merge_candidates(
        self,
        dense: list[dict[str, tp.Any]],
        lex: list[dict[str, tp.Any]],
    ) -> list[dict[str, tp.Any]]:
        import math

        def ffloat(v: tp.Any) -> float:
            """Безопасный float: ошибки/NaN/inf -> 0.0."""
            try:
                x = float(v)
                return x if math.isfinite(x) else 0.0
            except (TypeError, ValueError):
                return 0.0

        merged_dict: dict[str, dict[str, tp.Any]] = {}
        merge_key = self.settings.merge_by_field

        # 1) база из dense
        for d in dense:
            key_value = d.get(merge_key)
            if not key_value:
                self.logger.warning("Отсутствует ключ для слияния")
                continue
            x = merged_dict.setdefault(
                key_value,
                {
                    merge_key: key_value,
                    **{field: "" for field in self.settings.merge_fields},
                    **{score_field: 0.0 for score_field in ["score_dense", "score_lex"]},
                    "sources": set(),
                },
            )
            
            # дозаполняем тексты
            for fld in self.settings.merge_fields:
                if not x.get(fld) and d.get(fld):
                    x[fld] = d[fld]
            
            # агрегируем dense-скор (если вдруг встретится дубль)
            x["score_dense"] = max(ffloat(x.get("score_dense", 0.0)), ffloat(d.get("score_dense", 0.0)))
            x["sources"].add("dense")

        # 2) добавляем лексические кандидаты
        for l in lex:  # noqa: E741
            key_value = l.get(merge_key)
            if not key_value:
                self.logger.warning("Отсутствует ключ для слияния")
                continue
            x = merged_dict.setdefault(
                key_value,
                {
                    merge_key: key_value,
                    **{field: "" for field in self.settings.merge_fields},
                    **{score_field: 0.0 for score_field in ["score_dense", "score_lex"]},
                    "sources": set(),
                },
            )
            
            # дозаполняем тексты
            for fld in self.settings.merge_fields:
                if not x.get(fld) and l.get(fld):
                    x[fld] = l[fld]
            
            # агрегируем лучший лексический скор
            x["score_lex"] = max(ffloat(x.get("score_lex", 0.0)), ffloat(l.get("score_lex", 0.0)))
            # фиксация источника лексики
            x["sources"].add(l.get("_source", "lex"))

        # 3) приводим типы: числа -> float, sources -> list (детерминированный порядок)
        out: list[dict[str, tp.Any]] = []
        for x in merged_dict.values():
            x["score_dense"] = ffloat(x.get("score_dense", 0.0))
            x["score_lex"] = ffloat(x.get("score_lex", 0.0))
            if isinstance(x.get("sources"), set):
                x["sources"] = sorted(x["sources"])
            out.append(x)

        return out

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

    def _concat_text(self, item: dict[str, tp.Any]) -> str:
        fields = self.reranker_pairs_fields
        parts: list[str] = []

        for f in fields:
            val = item.get(f)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())

        return "\n\n".join(parts)


    async def warmup(self) -> bool:
        """Прогрев"""
        try:
            self.logger.info("🚩 Старт прогрева зависимостей")
            # 1) Redis доступ + запись с TTL (чтобы создать коннект и пул)
            await self.redis.set("__warmup__", "1", ttl=5)

            # 2) Эмбеддер → одно короткое кодирование (создаёт weights на устройстве и т.д.)
            vec = (await asyncio.to_thread(
                self.model.encode, ["warmup"], convert_to_numpy=True, normalize_embeddings=True
            ))[0]

            # 3) Milvus: «пустой» поиск на 1 документ (создаёт gRPC коннект/пул)
            with contextlib.suppress(Exception):
                await self.vector_db.search(
                    collection_name=self.settings.collection_name,
                    query_vector=vec,
                    top_k=1,
                )

            # 4) Лексическа
            if self.switches.use_hybrid:
                if self.switches.use_opensearch:
                    with contextlib.suppress(Exception):
                        await asyncio.to_thread(self.os_adapter.search, {"query": {"match_all": {}}}, 1)
                elif self.switches.use_bm25:
                    with contextlib.suppress(Exception):
                        await asyncio.to_thread(self.bm25.search, "warmup", 1)

            # 5) Реранкер — одно предсказание
            if self.switches.use_reranker:
                with contextlib.suppress(Exception):
                    _ = await asyncio.to_thread(self.ce.rank_fast, [("warmup", "warmup")])
            return True
        except Exception:
            # Прогрев не должен валить процесс — пишем в лог и идём дальше
            return False

    def _metrics_logger(self, label: str, start_time: float, precision: int = 4) -> float:
        """Фиксирует длительность и при включённом флаге логирует."""
        elapsed = time.perf_counter() - start_time
        elapsed_rounded = round(elapsed, precision)
        if self.log_metrics_enabled:
            self.logger.info(f"{label}: {elapsed_rounded:.{precision}f} сек")
        return elapsed_rounded

    def _metrics_value(self, label: str, value: float):
        """Выводит готовое значение метрики (если включено логирование)."""
        if self.log_metrics_enabled:
            self.logger.info(f"{label}: {value:.4f} сек")

    def _dense_to_unit(self, raw: float) -> float:
        """
        Приводит сырое значение dense-канала к [0,1], где 1 — лучше.
        Поведение управляется settings.hybrid.dense_metric.
        Допустимые значения:
          - "ip"               : inner product при normalize_embeddings=True → в [-1,1]
          - "cosine"           : явный cosine similarity в [-1,1]
          - "cosine_distance"  : distance = 1 - cosine ∈ [0,2]
          - "l2"               : (квадр.) L2-distance; для нормализованных векторов cos ≈ 1 - d/2
          - "unit"             : уже [0,1], ничего не делаем
        """
        import math

        m = self.dense_metric.lower()  # дефолт под normalize+IP
        x = float(raw)

        try:
            if m == "unit":
                s = x  # уже [0,1]
            elif m in ("ip", "cosine"):
                # cosine or dot of unit vectors: [-1, 1] → [0,1]
                s = (x + 1.0) / 2.0
            elif m == "cosine_distance":
                # distance = 1 - cos  ⇒  cos = 1 - dist  ⇒  [−1,1]
                cos = 1.0 - x
                s = (cos + 1.0) / 2.0
            elif m == "l2":
                # Для unit-векторов: squared L2 ≈ 2 - 2cos  ⇒  cos ≈ 1 - d/2
                # В Milvus/Faiss часто возвращается квадрат L2; формула устойчива и к нему.
                cos = 1.0 - (x / 2.0)
                s = (cos + 1.0) / 2.0
            else:
                s = x  # fallback, дальше зажмём
        except Exception:
            s = 0.0

        # жёстко зажимаем на всякий случай
        if not math.isfinite(s):
            s = 0.0
        return 0.0 if s < 0.0 else 1.0 if s > 1.0 else s

    def _precut_dense(
            self,
            items: list[dict],
            *,
            abs_min: float | None = None,
            rel_min: float | None = None,
            min_keep: int | None = None
    ) -> list[dict]:
        """ Оставляет только те dense-кандидаты, которые не хуже:
            - абсолютного минимума abs_min (в [0,1]),
            - относительного порога rel_min * top_dense.
        """
        if not items:
            return items

        abs_min = abs_min if abs_min is not None else self.settings.dense_abs_min
        rel_min = rel_min if rel_min is not None else self.settings.dense_rel_min
        min_keep = min_keep if min_keep is not None else self.settings.precut_min_keep

        top = max(float(x.get("score_dense", 0.0)) for x in items) or 1e-9
        thr = max(abs_min, rel_min * top)
        keep = [x for x in items if float(x.get("score_dense", 0.0)) >= thr]
        return keep if len(keep) >= min_keep else items[:min_keep]

    def _precut_lex(
            self,
            items: list[dict],
            *,
            rel_min: float | None = None,
            min_keep: int | None = None,
    ) -> list[dict]:
        """
        Нормализованная лексика: оставляем только те, что >= rel_min * top_lex.
        Предполагается, что score_lex уже приведён к [0,1] как raw/top.
        """
        if not items:
            return items


        rel_min = rel_min if rel_min is not None else self.settings.lex_rel_min
        min_keep = min_keep if min_keep is not None else self.settings.precut_min_keep

        top = max(float(x.get("score_lex", 0.0)) for x in items) or 1e-9
        thr = rel_min * top
        keep = [x for x in items if float(x.get("score_lex", 0.0)) >= thr]
        return keep if len(keep) >= min_keep else items[:min_keep]

