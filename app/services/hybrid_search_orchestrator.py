import asyncio
import contextlib
import json
import logging
import time
import traceback
import typing as tp
import uuid
from copy import deepcopy
from datetime import UTC, datetime

import pymorphy3
from sentence_transformers import SentenceTransformer

from app.common.logger import AISearchLogger
from app.common.storages.interfaces import KeyValueStorageProtocol
from app.domain.schemas.search_request import (
    SearchRequestCreateDTO,
    SearchRequestSchema,
)
from app.infrastructure.adapters.interfaces import (
    ICrossEncoderAdapter,
    IOpenSearchAdapter,
    IRedisSemaphore,
)
from app.infrastructure.adapters.light_interfaces import ILLMQueue
from app.infrastructure.storages.interfaces import IVectorDatabase
from app.infrastructure.unit_of_work.interfaces import IUnitOfWork
from app.infrastructure.utils.exact_filters import (
    ExactFilterConfig,
    NormalizedExactFilters,
    build_milvus_exact_filter_expr,
    build_opensearch_exact_filter_clauses,
    combine_milvus_filter_exprs,
    normalize_request_exact_filters,
)
from app.infrastructure.utils.metrics import _convert_to_ms_or_return_0, _now_ms
from app.infrastructure.utils.nlp import hash_query, normalize_query
from app.infrastructure.utils.search_cache_version import (
    build_search_cache_key,
    get_or_create_search_data_version,
)
from app.infrastructure.utils.token_filters import (
    MultiValueTokenConfig,
    NormalizedTokenFilters,
    build_milvus_token_filter_expr,
    build_opensearch_token_filter_clauses,
    normalize_request_token_filters,
)
from app.services.interfaces import IHybridSearchOrchestrator
from app.settings.config import Settings


class HybridSearchOrchestrator(IHybridSearchOrchestrator):
    """Оркестратор гибридного поиска"""

    def __init__(
        self,
        queue: ILLMQueue,
        sem: IRedisSemaphore,
        vector_db: IVectorDatabase,
        ce: ICrossEncoderAdapter,
        os_adapter: IOpenSearchAdapter,
        settings: Settings,
        redis: KeyValueStorageProtocol,
        logger: AISearchLogger,
        uow: IUnitOfWork,
    ):
        """Инициализирует зависимости и настройки pipeline гибридного поиска.

        Здесь же собирается `MultiValueTokenConfig`, который далее используется
        в `documents_search` для нормализации входных фильтров и построения
        backend-специфичных условий (OpenSearch/Milvus).
        """
        self.queue = queue
        self.sem = sem
        self.vector_db = vector_db

        self.ce = ce
        self.os_adapter = os_adapter
        self.redis = redis
        self.settings = settings.hybrid
        self.switches = settings.switches
        self.ce_settings = settings.reranker
        self.short = settings.short_settings
        self.model = SentenceTransformer(
            settings.milvus.model_name, local_files_only=True
        )
        self.morph = pymorphy3.MorphAnalyzer()
        self.model_name = settings.milvus.model_name.split("/")[-1]
        self.ce_model_name = self.ce_settings.model_name.split("/")[-1]
        self.normalize_query = settings.app.normalize_query
        self.os_index_name = settings.opensearch.index_name
        self.intermediate_results_top_k = settings.hybrid.intermediate_results_top_k
        self.log_metrics_enabled = settings.search_metrics.log_metrics_enabled
        self.response_metrics_enabled = settings.search_metrics.response_metrics_enabled
        self.token_filter_config = MultiValueTokenConfig(
            raw_fields=settings.token_filters.raw_fields,
            token_suffix=settings.token_filters.token_suffix,
            raw_separator=settings.token_filters.raw_separator,
        )

        self.exact_filter_config = ExactFilterConfig(
            raw_fields=settings.exact_filters.raw_fields,
            field_suffix=settings.exact_filters.field_suffix,
        )

        self.dense_metric = settings.milvus.metric_type
        self.reranker_pairs_fields = settings.reranker.pairs_fields
        self.logger = logger

        self.uow: IUnitOfWork = uow

    def set_logger(self, logger: logging.Logger) -> None:
        """Переопределение логгера"""
        self.logger = logger  # type: ignore

    async def documents_search(
        self,
        task_id: str,
        ticket_id: str,
        pack_key: str,
        result_key: str,
    ) -> dict[str, str]:
        """Запускает полный pipeline гибридного поиска по задаче из очереди.

        Ключевые этапы: чтение pack из Redis, нормализация query/filters,
        presearch, dense+lex поиск, merge/rerank, кеширование и запись ответа.
        Token-фильтры применяются только в основном dense/lex pipeline:
        - векторный поиск (Milvus `filter_expr`);
        - lexical поиск (OpenSearch `bool.filter`);
        - cache key итогового ответа (детерминированная часть по фильтрам).

        Presearch — отдельный exact-match этап по `presearch_field`, который
        выполняется без token-фильтрации.
        """
        start_total = time.perf_counter()
        metrics: dict[str, float] = {}

        queued_at = await self.redis.hash_get(
            f"{self.queue.tprefix}{ticket_id}", "queued_at_ms"
        )

        await self.queue.set_running(ticket_id, task_id)

        # ---- Redis GET pack ----
        redis_get_start = time.perf_counter()
        raw = await self.redis.get(pack_key)
        metrics["redis_get_time"] = self._metrics_logger(
            "🕒 Redis GET pack", redis_get_start
        )

        if not raw:
            await self.queue.set_failed(ticket_id, "missing pack")
            await self.queue.ack(ticket_id)
            return {"status": "error", "error": "missing pack"}

        # ---- JSON parse ----
        json_parse_start = time.perf_counter()
        pack = json.loads(raw)
        metrics["json_parse_time"] = self._metrics_logger(
            "🕒 JSON parse", json_parse_start
        )

        settings_local = deepcopy(self.settings)
        switches_local = deepcopy(self.switches)

        raw_query = str(pack["query"]).strip()
        # Внешний контракт pack использует array_filters; внутри сохраняем
        # существующий термин token_filters как нормализованное представление.
        raw_array_filters = pack.get("array_filters") or {}
        raw_filters: dict[str, list[str] | None] = {
            raw_field: raw_array_filters.get(raw_field)
            for raw_field in self.token_filter_config.raw_fields
        }
        # Нормализация token-фильтров в терминах token-полей (role_tokens и т.п.).
        token_filters = normalize_request_token_filters(
            raw_filters, config=self.token_filter_config
        )
        self.logger.debug(f"Normalized token filters: {token_filters.by_token_field}")

        raw_exact_filters = pack.get("exact_filters") or {}
        exact_filters = normalize_request_exact_filters(
            raw_exact_filters,
            config=self.exact_filter_config,
        )
        self.logger.debug(f"Normalized exact filters: {exact_filters.by_filter_field}")

        token_filter_cache_key = token_filters.cache_key_part()
        exact_filter_cache_key = exact_filters.cache_key_part()
        filter_cache_key = (
            f"tokens:{token_filter_cache_key};exact:{exact_filter_cache_key}"
        )

        milvus_token_filter_expr = build_milvus_token_filter_expr(token_filters)
        milvus_exact_filter_expr = build_milvus_exact_filter_expr(exact_filters)
        milvus_filter_expr = combine_milvus_filter_exprs(
            milvus_token_filter_expr,
            milvus_exact_filter_expr,
        )
        # ---- Short settings ----
        if self.short.mode:
            sh_start = time.perf_counter()

            query_tokens = normalize_query(query=raw_query, morph=self.morph).split()
            self._metrics_logger(
                label="🕒 Normalize query (short mode)", start_time=sh_start
            )
            # self.logger.info(f"query: {query_tokens}")

            if len(query_tokens) <= self.short.mode_limit:
                override_start = time.perf_counter()
                self.logger.info("Используем short настройки для запроса")

                settings_local.top_k = self.short.top_k
                settings_local.w_lex = self.short.w_lex
                settings_local.w_dense = self.short.w_dense
                settings_local.dense_top_k = self.short.dense_top_k
                settings_local.w_ce = self.short.w_ce
                settings_local.lex_top_k = self.short.lex_top_k

                switches_local = deepcopy(self.short)
                self._metrics_logger(
                    label="🕒 Override (short mode)", start_time=override_start
                )

            else:
                self.logger.info("Используем обычные настройки для запроса")

        # ---- Normalize query ----
        if self.normalize_query:
            normalize_start = time.perf_counter()
            query = normalize_query(query=raw_query, morph=self.morph)
            query_hash = hash_query(query)
            top_k = int(pack["top_k"])
            metrics["normalize_time"] = self._metrics_logger(
                "🕒 Normalize query", normalize_start
            )
        else:
            query = raw_query
            top_k = int(pack["top_k"])
            query_hash = raw_query

        need_ack = True

        if switches_local.use_opensearch:
            lex_candidate = "OpenSearch"

        else:
            lex_candidate = ""
        w_dense = settings_local.w_dense
        w_lex = settings_local.w_lex if switches_local.use_hybrid else 0.0
        w_ce = settings_local.w_ce if switches_local.use_reranker else 0.0

        lex_enable = bool(w_lex)
        reranker_enable = bool(w_ce)
        dense, lex = [], []

        try:
            async with self.sem.acquire(), self.uow:
                query_start_time = datetime.now(UTC)
                metrics["semaphore_acquire_time"] = self._metrics_logger(
                    "🕒 Semaphore acquire", time.perf_counter()
                )

                search_use_cache = bool(pack.get("search_use_cache", True))
                show_intermediate_results = bool(pack.get("show_intermediate_results", False))
                presearch_config = pack.get("presearch") or None

                presearch_field = None
                presearch_enabled = False
                if isinstance(presearch_config, dict):
                    raw_presearch_field = presearch_config.get("field")
                    if raw_presearch_field is not None:
                        presearch_field = str(raw_presearch_field).strip()
                        presearch_enabled = bool(presearch_field)

                if presearch_enabled and presearch_field:
                    presearch_key_part = f"1:{presearch_field}:{hash_query(raw_query)}"
                else:
                    presearch_key_part = "0:no_presearch"

                data_version = await get_or_create_search_data_version(
                    self.redis,
                    collection_name=settings_local.collection_name,
                    index_name=self.os_index_name,
                )
                hybrid_version = (
                    f"{settings_local.version}"
                    f":fusion={settings_local.fusion_mode}"
                    f":rrf_k={settings_local.rrf_k}"
                )
                cache_key = build_search_cache_key(
                    collection_name=settings_local.collection_name,
                    index_name=self.os_index_name,
                    data_version=data_version,
                    hybrid_version=hybrid_version,
                    query_hash=query_hash,
                    top_k=top_k,
                    presearch_key_part=presearch_key_part,
                    filters_key_part=filter_cache_key,
                )

                # show_intermediate_results требует свежего исполнения pipeline,
                # так как intermediate_results не хранятся в legacy result-cache.
                allow_cache_read = search_use_cache and not show_intermediate_results
                cached = await self.redis.get(cache_key) if allow_cache_read else None

                if show_intermediate_results:
                    self.logger.debug(
                        "Cache read skipped because show_intermediate_results=True requires fresh pipeline execution"
                    )
                elif not search_use_cache:
                    self.logger.debug(
                        "Cache read skipped because request search_use_cache=False"
                    )

                if cached:
                    self.logger.info("📦 Выдаем результат из кеша")
                    cache_parse_start = time.perf_counter()
                    results = json.loads(cached)
                    metrics["cache_parse_time"] = self._metrics_logger(
                        "🕒 Cache parse", cache_parse_start
                    )
                    merged = None
                else:
                    # ---- Presearch ----
                    presearch_result = None
                    if presearch_enabled and presearch_field:
                        presearch_start = time.perf_counter()
                        presearch_result = await self._presearch_exact_match(
                            query=raw_query,
                            field_name=presearch_field,
                            use_ce=switches_local.use_reranker,
                        )
                        metrics["presearch_time"] = self._metrics_logger(
                            "🕒 Presearch (exact match)", presearch_start
                        )

                    # ---- Embedding ----
                    encode_start = time.perf_counter()
                    query_vector = (
                        await asyncio.to_thread(
                            self.model.encode,
                            [query],
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                        )
                    )[0]
                    metrics["embedding_time"] = self._metrics_logger(
                        "🕒 Model encode", encode_start
                    )

                    # ---- Dense & Lex search ----
                    async def _dense_task() -> list[dict[str, tp.Any]]:
                        if w_dense == 0.0:
                            return []
                        start = time.perf_counter()
                        res = await self.vector_db.search(
                            collection_name=settings_local.collection_name,
                            query_vector=query_vector,
                            top_k=settings_local.dense_top_k,
                            filter_expr=milvus_filter_expr,
                        )
                        for d in res:
                            d["score_dense"] = self._dense_to_unit(
                                d.get("score_dense", 0.0)
                            )
                        res = self._precut_dense(res)
                        metrics["vector_search_time"] = self._metrics_logger(
                            "🕒 Milvus search", start
                        )
                        return res

                    async def _lex_task() -> list[dict[str, tp.Any]]:
                        if w_lex == 0.0:
                            return []
                        start = time.perf_counter()
                        if switches_local.use_hybrid:
                            if switches_local.use_opensearch:
                                res = await self._os_candidates(
                                    query=query,
                                    k=settings_local.lex_top_k,
                                    token_filters=token_filters,
                                    exact_filters=exact_filters,
                                )
                            else:
                                res = []
                        else:
                            res = []
                        res = self._precut_lex(res)
                        metrics["lexical_search_time"] = self._metrics_logger(
                            "🕒 Lexical search", start
                        )
                        return res

                    dense, lex = await asyncio.gather(_dense_task(), _lex_task())

                    if settings_local.fusion_mode == "rrf":
                        merged = self._merge_candidates_rrf(
                            dense=dense,
                            lex=lex,
                            w_dense=w_dense,
                            w_lex=w_lex,
                            rrf_k=settings_local.rrf_k,
                        )
                    else:
                        merged = self._merge_candidates(dense, lex)
                        for item in merged:
                            item["score_fusion"] = (
                                w_dense * float(item.get("score_dense", 0.0))
                                + w_lex * float(item.get("score_lex", 0.0))
                            )

                    # ---- Cross-encoder final rerank stage ----
                    reranker_enabled = bool(switches_local.use_reranker and w_ce > 0.0)
                    if reranker_enabled and merged:
                        start = time.perf_counter()
                        pairs = [(query, self._concat_text(m)) for m in merged]

                        start_rank_fast = time.perf_counter()
                        scores = await asyncio.to_thread(self.ce.rank_fast, pairs)
                        self._metrics_logger(
                            label="🕒 Encoder (rank_fast)", start_time=start_rank_fast
                        )

                        start_postprocess = time.perf_counter()
                        scores = self.ce.ce_postprocess(scores)
                        self._metrics_logger(
                            label="🕒 Encoder (postprocess)",
                            start_time=start_postprocess,
                        )

                        metrics["cross_encoder_time"] = self._metrics_logger(
                            "🕒 Cross-encoder rank", start
                        )
                        for m, s in zip(merged, scores, strict=True):
                            m["score_ce"] = float(s)

                    _results = self._score_and_slice(
                        merged,
                        top_k,
                        use_ce=switches_local.use_reranker,
                        w_dense=w_dense,
                        w_lex=w_lex,
                        w_ce=w_ce,
                        fusion_mode=settings_local.fusion_mode,
                    )
                    if presearch_result:
                        _results = self._inject_presearch_result(
                            results=_results,
                            presearch_result=presearch_result,
                            top_k=top_k,
                            merge_by_field=settings_local.merge_by_field,
                        )

                    results = _results

                    await self.redis.set(
                        cache_key,
                        json.dumps(results, ensure_ascii=False),
                        ttl=settings_local.cache_ttl,
                    )

                metrics["total_search_time"] = self._metrics_logger(
                    "🕒 Total search", start_total
                )

                payload = {"results": results}
                if show_intermediate_results:
                    intermediate_results: dict[str, list[dict[str, tp.Any]]] = {}

                    if dense:
                        intermediate_results["dense"] = sorted(
                            dense, key=lambda x: x.get("score_dense", 0.0), reverse=True
                        )[: self.intermediate_results_top_k]
                    else:
                        intermediate_results["dense"] = []

                    if lex:
                        intermediate_results["lex"] = sorted(
                            lex, key=lambda x: x.get("score_lex", 0.0), reverse=True
                        )[: self.intermediate_results_top_k]
                    else:
                        intermediate_results["lex"] = []
                    if merged:
                        intermediate_results["fusion"] = sorted(
                            merged, key=lambda x: x.get("score_fusion", 0.0), reverse=True
                        )[: self.intermediate_results_top_k]
                    else:
                        intermediate_results["fusion"] = []
                    if merged and reranker_enabled:
                        intermediate_results["ce"] = sorted(
                            merged, key=lambda x: x.get("score_ce", 0.0), reverse=True
                        )[: self.intermediate_results_top_k]
                    else:
                        intermediate_results["ce"] = []

                    payload["intermediate_results"] = intermediate_results
                metrics["full_search_task_time"] = _now_ms() - int(queued_at)
                search_request: SearchRequestSchema = (
                    await self.uow.search_request.create(
                        create_dto=SearchRequestCreateDTO(
                            id=uuid.uuid4(),
                            query=pack["query"],
                            search_start_time=query_start_time,
                            full_execution_time=_convert_to_ms_or_return_0(
                                metrics.get("full_search_task_time")
                            ),
                            search_execution_time=_convert_to_ms_or_return_0(
                                metrics.get("total_search_time")
                            ),
                            dense_search_time=_convert_to_ms_or_return_0(
                                metrics.get("vector_search_time")
                            ),
                            lex_search_time=_convert_to_ms_or_return_0(
                                metrics.get("lexical_search_time")
                            ),
                            query_norm_time=_convert_to_ms_or_return_0(
                                metrics.get("normalize_time")
                            ),
                            reranker_time=_convert_to_ms_or_return_0(
                                metrics.get("cross_encoder_time")
                            ),
                            model_name=self.model_name,
                            reranker_name=self.ce_model_name,
                            reranker_enable=reranker_enable,
                            lex_enable=lex_enable,
                            from_cache=bool(metrics.get("cache_parse_time")),
                            lex_candidate=lex_candidate,
                            dense_top_k=len(dense),
                            lex_top_k=len(lex),
                            top_k=top_k,
                            weight_ce=settings_local.w_ce,
                            weight_dense=settings_local.w_dense,
                            weight_lex=settings_local.w_lex,
                            results=results,
                        )
                    )
                )
                payload["search_request_id"] = str(search_request.id)

                if self.response_metrics_enabled:
                    payload["metrics"] = {
                        "presearch_time": metrics.get("presearch_time"),
                        "presearch_enabled": presearch_enabled,
                        "search_use_cache": search_use_cache,
                        "cache_read_allowed": allow_cache_read,
                        "show_intermediate_results": show_intermediate_results,
                        "presearch_field": presearch_field or "",
                        "embedding_time": metrics.get("embedding_time"),
                        "vector_search_time": metrics.get("vector_search_time"),
                        "lexical_search_time": metrics.get("lexical_search_time"),
                        "full_search_task_time": metrics.get("full_search_task_time"),
                        "cross_encoder_time": metrics.get("cross_encoder_time"),
                        "total_time": metrics.get("total_search_time"),
                        "reranker_enabled": reranker_enable,
                        "open_search_enabled": lex_enable,
                        "from_cache": metrics.get("cache_parse_time", False),
                        "hybrid_dense_top_k": len(dense),
                        "hybrid_lex_top_k": len(lex),
                        "hybrid_top_k": top_k,
                        "hybrid_w_ce": w_ce,
                        "hybrid_w_dense": w_dense,
                        "hybrid_w_lex": w_lex,
                        "hybrid_fusion_mode": settings_local.fusion_mode,
                        "hybrid_rrf_k": settings_local.rrf_k,
                        "hybrid_score_final_mode": (
                            "cross_encoder_final" if reranker_enabled and merged else "fusion_only"
                        ),
                        "encoder_model": self.model_name,
                        "reranker_model": self.ce_model_name,
                    }

                await self.redis.set(
                    result_key,
                    json.dumps(payload, ensure_ascii=False),
                    ttl=self.queue.ticket_ttl,
                )
                await self.queue.set_done(ticket_id)

                await self.uow.commit()

                return {"status": "ok"}

        finally:
            if need_ack:
                ack_start = time.perf_counter()
                with contextlib.suppress(Exception):
                    await self.queue.ack(ticket_id)
                metrics["queue_ack_time"] = self._metrics_logger(
                    "🕒 Queue ACK", ack_start
                )

    async def _os_candidates(
        self,
        query: str,
        k: int,
        token_filters: NormalizedTokenFilters,
        exact_filters: NormalizedExactFilters,
    ) -> list[dict[str, tp.Any]]:
        """Получает lexical-кандидатов из OpenSearch с учётом token-фильтров.

        Используется внутри `documents_search` как lexical ветка гибридного
        поиска. Фильтры передаются через `build_opensearch_token_filter_clauses`
        с семантикой OR внутри поля и AND между полями.
        """
        os_adapter = self.os_adapter
        filter_clauses = [
            *build_opensearch_token_filter_clauses(token_filters),
            *build_opensearch_exact_filter_clauses(exact_filters),
        ]
        body = {
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": os_adapter.config.search_fields,
                            "operator": os_adapter.config.operator,
                            "fuzziness": os_adapter.config.fuzziness,
                        }
                    },
                    "filter": filter_clauses,
                }
            }
        }

        hits = await os_adapter.search(body, k)

        out: list[dict[str, tp.Any]] = []
        output_fields = os_adapter.config.output_fields

        for h in hits:
            src = h.get("_source", {})
            raw = float(h.get("_score", 0.0))  # сырое значение от OpenSearch

            result_item = {
                "score_lex_raw": raw,  # сохраняем «как есть»
                "score_lex": raw,  # пока сырое — нормализуем ниже
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

    async def _presearch_exact_match(
        self,
        query: str,
        field_name: str,
        *,
        use_ce: bool,
    ) -> dict[str, tp.Any] | None:
        """Ищет один точный (без учета регистра) результат по заданному полю OpenSearch.

        Используется в `documents_search` перед основным dense/lex этапом.
        Этот шаг намеренно НЕ использует token-фильтры role/product: presearch
        должен находить exact-match документ независимо от фильтров основного поиска.

        Важный момент: в разных схемах индекса `term + case_insensitive` может
        срабатывать только на части полей, поэтому используем двухэтапный подход:
        1) быстрый exact-запрос term/case_insensitive по `<field>` и `<field>.keyword`;
        2) fallback через match_phrase + строгая проверка в Python
           (`value.casefold() == query.casefold()`), чтобы гарантировать
           регистронезависимое точное совпадение по исходной строке поля.
        """
        os_adapter = self.os_adapter
        output_fields = os_adapter.config.output_fields

        query_folded = query.casefold()

        # 1) Наиболее надежный путь: script query по doc-values (обычно keyword).
        # Не зависит от поддержки `case_insensitive` в term-query для конкретной версии OS.
        script_should = [
            {
                "script": {
                    "script": {
                        "lang": "painless",
                        "source": (
                            "if (!doc.containsKey(params.field) || doc[params.field].empty) "
                            "{ return false; } "
                            "return doc[params.field].value.toString().toLowerCase() == params.q;"
                        ),
                        "params": {"field": field_name, "q": query_folded},
                    }
                }
            },
            {
                "script": {
                    "script": {
                        "lang": "painless",
                        "source": (
                            "if (!doc.containsKey(params.field) || doc[params.field].empty) "
                            "{ return false; } "
                            "return doc[params.field].value.toString().toLowerCase() == params.q;"
                        ),
                        "params": {"field": f"{field_name}.keyword", "q": query_folded},
                    }
                }
            },
        ]
        body = {
            "query": {
                "bool": {
                    "minimum_should_match": 1,
                    "should": script_should,
                }
            }
        }
        hits = await os_adapter.search(body, 1)

        if hits:
            for hit in hits:
                if self._is_case_insensitive_exact_match(
                    source=hit.get("_source", {}),
                    field_name=field_name,
                    query=query,
                ):
                    return self._build_presearch_item(hit, output_fields, use_ce=use_ce)

        # 1.1) Дополнительный быстрый fallback на случай отключенных скриптов.
        should_terms = [
            {"term": {field_name: {"value": query, "case_insensitive": True}}},
            {
                "term": {
                    f"{field_name}.keyword": {"value": query, "case_insensitive": True}
                }
            },
        ]
        term_body = {
            "query": {
                "bool": {
                    "minimum_should_match": 1,
                    "should": should_terms,
                }
            }
        }
        term_hits = await os_adapter.search(term_body, 1)
        for hit in term_hits:
            if self._is_case_insensitive_exact_match(
                source=hit.get("_source", {}),
                field_name=field_name,
                query=query,
            ):
                return self._build_presearch_item(hit, output_fields, use_ce=use_ce)

        # Fallback: расширяем окно кандидатов и фильтруем exact-match локально.
        fallback_body = {
            "query": {
                "bool": {"must": {"match_phrase": {field_name: {"query": query}}}}
            }
        }
        fallback_hits = await os_adapter.search(fallback_body, 20)
        for hit in fallback_hits:
            if self._is_case_insensitive_exact_match(
                source=hit.get("_source", {}),
                field_name=field_name,
                query=query,
            ):
                return self._build_presearch_item(hit, output_fields, use_ce=use_ce)

        return None

    def _is_case_insensitive_exact_match(
        self,
        source: dict[str, tp.Any],
        field_name: str,
        query: str,
    ) -> bool:
        """Строгое сравнение строки поля и query без учета регистра.

        Запрос уже приходит в виде `strip()` из внешней логики.
        Содержимое поля не трогаем (не strip), чтобы сохранить требование
        «точное совпадение символов, кроме регистра».
        """
        value = source.get(field_name)
        if value is None:
            return False
        if isinstance(value, str):
            value_str = value
        else:
            value_str = str(value)
        return value_str.casefold() == query.casefold()

    def _build_presearch_item(
        self,
        hit: dict[str, tp.Any],
        output_fields: list[str],
        *,
        use_ce: bool,
    ) -> dict[str, tp.Any]:
        """Приводит hit предварительного поиска к единому формату выдачи."""
        src = hit.get("_source", {})
        item: dict[str, tp.Any] = {
            "score_lex_raw": float(hit.get("_score", 0.0)),
            "score_dense": 0.0,
            "score_lex": 1.0,
            "score_final": 1.0,
            "_source": "presearch",
            "sources": ["presearch"],
        }
        if use_ce:
            item["score_ce"] = 0.0

        for field in output_fields:
            item[field] = src.get(field, "")

        return item

    def _inject_presearch_result(
        self,
        results: list[dict[str, tp.Any]],
        presearch_result: dict[str, tp.Any],
        top_k: int,
        merge_by_field: str,
    ) -> list[dict[str, tp.Any]]:
        """Встраивает presearch-результат на первое место.

        Если результат уже есть в выдаче, удаляем дубликат по merge-полю.
        Затем добавляем presearch наверх и подрезаем до top_k.
        """
        result_key_value = presearch_result.get(merge_by_field)
        filtered_results = [
            item
            for item in results
            if not (
                result_key_value
                and item.get(merge_by_field)
                and item.get(merge_by_field) == result_key_value
            )
        ]

        merged_results = [presearch_result, *filtered_results]
        return merged_results[:top_k]

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
                    **dict.fromkeys(self.settings.merge_fields, ""),
                    **dict.fromkeys(["score_dense", "score_lex"], 0.0),
                    "sources": set(),
                },
            )

            # дозаполняем тексты
            for fld in self.settings.merge_fields:
                if not x.get(fld) and d.get(fld):
                    x[fld] = d[fld]

            # агрегируем dense-скор (если вдруг встретится дубль)
            x["score_dense"] = max(
                ffloat(x.get("score_dense", 0.0)), ffloat(d.get("score_dense", 0.0))
            )
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
                    **dict.fromkeys(self.settings.merge_fields, ""),
                    **dict.fromkeys(["score_dense", "score_lex"], 0.0),
                    "sources": set(),
                },
            )

            # дозаполняем тексты
            for fld in self.settings.merge_fields:
                if not x.get(fld) and l.get(fld):
                    x[fld] = l[fld]

            # агрегируем лучший лексический скор
            x["score_lex"] = max(
                ffloat(x.get("score_lex", 0.0)), ffloat(l.get("score_lex", 0.0))
            )
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

    def _merge_candidates_rrf(
        self,
        dense: list[dict[str, tp.Any]],
        lex: list[dict[str, tp.Any]],
        *,
        w_dense: float,
        w_lex: float,
        rrf_k: int,
    ) -> list[dict[str, tp.Any]]:
        """Объединяет dense/lex-кандидатов через RRF.

        Метод сохраняет текущую семантику merge по merge_by_field/merge_fields и
        дополняет её score_rrf_raw + score_fusion (нормализованный RRF).
        """
        merged = self._merge_candidates(dense, lex)
        merge_key = self.settings.merge_by_field
        dense_sorted = sorted(dense, key=lambda x: float(x.get("score_dense", 0.0)), reverse=True)
        lex_sorted = sorted(lex, key=lambda x: float(x.get("score_lex", 0.0)), reverse=True)
        rrf_scores: dict[str, float] = {}

        for rank, item in enumerate(dense_sorted, start=1):
            key = item.get(merge_key)
            if key:
                rrf_scores[key] = rrf_scores.get(key, 0.0) + (w_dense / (rrf_k + rank))

        for rank, item in enumerate(lex_sorted, start=1):
            key = item.get(merge_key)
            if key:
                rrf_scores[key] = rrf_scores.get(key, 0.0) + (w_lex / (rrf_k + rank))

        max_rrf = max(rrf_scores.values(), default=0.0)
        for item in merged:
            key = item.get(merge_key)
            raw_rrf = float(rrf_scores.get(key, 0.0))
            item["score_rrf_raw"] = raw_rrf
            item["score_fusion"] = (raw_rrf / max_rrf) if max_rrf > 0.0 else 0.0
        return merged

    def _score_and_slice(
        self,
        items: list[dict[str, tp.Any]],
        top_k: int,
        *,
        use_ce: bool,
        w_dense: float,
        w_lex: float,
        w_ce: float,
        fusion_mode: str,
    ) -> list[dict[str, tp.Any]]:
        """Формирует score_final и сортирует выдачу.

        Для weighted_score score_fusion вычисляется как взвешенная сумма dense/lex.
        Для rrf score_fusion ожидается из _merge_candidates_rrf.
        Cross-encoder, если реально включен, всегда выступает финальным сортировщиком.
        """
        if not items:
            return []

        for it in items:
            if fusion_mode == "weighted_score" and "score_fusion" not in it:
                it["score_fusion"] = (
                    w_dense * float(it.get("score_dense", 0.0))
                    + w_lex * float(it.get("score_lex", 0.0))
                )
            elif fusion_mode == "rrf":
                it["score_fusion"] = float(it.get("score_fusion", 0.0))

        effective_use_ce = bool(
            use_ce and w_ce > 0.0 and any("score_ce" in item for item in items)
        )
        if effective_use_ce:
            for it in items:
                it["score_final"] = float(it.get("score_ce", 0.0))
            items.sort(
                key=lambda x: (x.get("score_ce", 0.0), x.get("score_fusion", 0.0)),
                reverse=True,
            )
        else:
            for it in items:
                it["score_final"] = float(it.get("score_fusion", 0.0))
            items.sort(key=lambda x: x.get("score_fusion", 0.0), reverse=True)
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
            vec = (
                await asyncio.to_thread(
                    self.model.encode,
                    ["warmup"],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            )[0]

            # 3) Milvus: «пустой» поиск на 1 документ (создаёт gRPC коннект/пул)
            await self.vector_db.search(
                collection_name=self.settings.collection_name,
                query_vector=vec,
                top_k=1,
            )

            # 4) Лексическа
            if self.switches.use_hybrid:
                if self.switches.use_opensearch:
                    await self.os_adapter.search({"query": {"match_all": {}}}, 1)

            # 5) Реранкер — одно предсказание
            if self.switches.use_reranker:
                _ = await asyncio.to_thread(self.ce.rank_fast, [("warmup", "warmup")])
            # 6) normalize
            normalize_query("врач", self.morph)
            return True
        except Exception as e:
            self.logger.error(f"Ошибка прогрева: ({type(e)}): {traceback.format_exc()}")
            return False

    def _metrics_logger(
        self, label: str, start_time: float, precision: int = 4
    ) -> float:
        """Фиксирует длительность и при включённом флаге логирует."""
        elapsed = time.perf_counter() - start_time
        elapsed_rounded = round(elapsed, precision)
        if self.log_metrics_enabled:
            self.logger.info(f"{label}: {elapsed_rounded:.{precision}f} сек")
        return elapsed_rounded

    def _metrics_value(self, label: str, value: float) -> None:
        """Выводит готовое значение метрики (если включено логирование)."""
        if self.log_metrics_enabled:
            self.logger.info(f"{label}: {value:.4f} сек")

    def _dense_to_unit(self, raw: float) -> float:
        """Приводит сырое значение dense-канала к [0,1], где 1 — лучше.
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
        min_keep: int | None = None,
    ) -> list[dict]:
        """Оставляет только те dense-кандидаты, которые не хуже:
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
        """Нормализованная лексика: оставляем только те, что >= rel_min * top_lex.
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
