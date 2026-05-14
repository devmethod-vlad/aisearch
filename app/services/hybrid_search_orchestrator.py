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
        self.model = SentenceTransformer(settings.milvus.model_name, local_files_only=True)
        self.morph = pymorphy3.MorphAnalyzer()
        self.model_name = settings.milvus.model_name.split("/")[-1]
        self.ce_model_name = self.ce_settings.model_name.split("/")[-1]
        self.normalize_query = settings.app.normalize_query
        self.os_index_name = settings.opensearch.index_name
        self.intermediate_results_top_k = settings.hybrid.intermediate_results_top_k
        self.log_metrics_enabled = settings.search_metrics.log_metrics_enabled
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

    def _extract_array_filters(
        self,
        pack: dict[str, tp.Any],
    ) -> dict[str, list[str] | None]:
        """Извлекает и валидирует контейнер `array_filters` из search pack.

        Функция используется в `documents_search` перед вызовом
        `normalize_request_token_filters`, чтобы защитить pipeline от
        некорректного payload из Redis (например, строки или списка вместо dict).
        Допустимая форма входа:
        - отсутствующее поле / `null` -> `{}`;
        - объект (`dict`) с ключами фильтров и значениями `list` или `null`.
        """
        raw_array_filters = pack.get("array_filters")
        if raw_array_filters is None:
            return {}
        if not isinstance(raw_array_filters, dict):
            raise ValueError(
                "invalid array_filters: expected object/dict with filter names as keys and lists of values as values"
            )

        for field_name, field_values in raw_array_filters.items():
            if field_values is not None and not isinstance(field_values, list):
                raise ValueError(f"invalid array_filters.{field_name}: expected list of values or null")
        return tp.cast(dict[str, list[str] | None], raw_array_filters)

    def _extract_exact_filters(
        self,
        pack: dict[str, tp.Any],
    ) -> dict[str, tp.Any]:
        """Извлекает и валидирует контейнер `exact_filters` из search pack.

        Функция используется в `documents_search` перед вызовом
        `normalize_request_exact_filters`, чтобы исключить падения на попытке
        обращения к `.get(...)` у не-словаря. Валидируется только тип контейнера:
        отсутствующее поле / `null` трактуется как `{}`, `dict` принимается как есть.
        """
        raw_exact_filters = pack.get("exact_filters")
        if raw_exact_filters is None:
            return {}
        if not isinstance(raw_exact_filters, dict):
            raise ValueError("invalid exact_filters: expected object/dict with filter names as keys")
        return raw_exact_filters

    def _resolve_top_k_for_fresh_search(
        self,
        pack: dict[str, tp.Any],
        settings_local: tp.Any,
    ) -> int:
        """Возвращает effective `top_k` для нового pipeline-поиска.

        Если `top_k` явно передан в request/pack — используется он.
        Иначе берётся текущее значение `settings_local.top_k` (включая short-mode
        override, если он уже применён).
        """
        requested_top_k = pack.get("top_k")
        if requested_top_k is not None:
            return int(requested_top_k)
        return int(settings_local.top_k)

    def _resolve_top_k_for_cached_response(
        self,
        pack: dict[str, tp.Any],
        settings_local: tp.Any,
    ) -> int:
        """Возвращает `top_k` для cached-response с fallback для legacy pack."""
        cached_top_k = pack.get("top_k")
        if cached_top_k is not None:
            return int(cached_top_k)
        return int(settings_local.top_k)

    def _dump_search_cache_payload(
        self,
        *,
        results: list[dict[str, tp.Any]],
        intermediate_results: dict[str, tp.Any] | None = None,
    ) -> str:
        """Сериализует search-cache payload только в новой object-схеме.

        Функция используется в `documents_search` перед записью в Redis и
        гарантирует единый формат cache entry:
        - обязательное поле `results` всегда присутствует;
        - поле `intermediate_results` добавляется только при переданном значении;
        - legacy list-формат и служебный `schema_version` не используются.
        """
        payload: dict[str, tp.Any] = {"results": results}
        if intermediate_results is not None:
            payload["intermediate_results"] = intermediate_results
        return json.dumps(payload, ensure_ascii=False)

    def _load_search_cache_payload(
        self,
        raw: str,
    ) -> tuple[list[dict[str, tp.Any]], dict[str, tp.Any] | None]:
        """Десериализует search-cache payload с безопасным fallback для legacy.

        Используется при cache read внутри `documents_search`:
        - новый формат: dict с `results` и опциональным `intermediate_results`;
        - legacy формат: list результатов (поддерживается только для чтения);
        - неизвестный/битый формат: поднимается `ValueError`, чтобы обработать
          entry как контролируемый cache miss, не падая всем поиском.
        """
        data = json.loads(raw)
        if isinstance(data, dict):
            results = data.get("results")
            if not isinstance(results, list):
                raise ValueError("invalid search-cache payload: 'results' must be a list")
            intermediate_results = data.get("intermediate_results")
            if not isinstance(intermediate_results, dict):
                intermediate_results = None
            return results, intermediate_results
        if isinstance(data, list):
            self.logger.warning("Legacy search-cache list format was read; use object payload")
            return data, None
        raise ValueError("invalid search-cache payload: expected object or list")

    def _apply_short_settings(
        self,
        settings_local: tp.Any,
        switches_local: tp.Any,
    ) -> tuple[tp.Any, tp.Any]:
        """Применяет short-mode override к effective search-настройкам.

        Функция используется в `documents_search`, когда short-mode действительно
        сработал (`SHORT_MODE=true` и длина запроса <= `SHORT_MODE_LIMIT`).
        Обновляет retrieval/fusion/final-score параметры на short-значения.
        Важно: `HYBRID_FINAL_RANK_MODE` остаётся глобальным и здесь не
        переопределяется.
        """
        settings_local.top_k = self.short.top_k
        settings_local.w_lex = self.short.w_lex
        settings_local.w_dense = self.short.w_dense
        settings_local.rrf_w_dense = self.short.rrf_w_dense
        settings_local.rrf_w_lex = self.short.rrf_w_lex
        settings_local.dense_top_k = self.short.dense_top_k
        settings_local.lex_top_k = self.short.lex_top_k
        settings_local.fusion_mode = self.short.fusion_mode
        settings_local.rrf_k = self.short.rrf_k
        settings_local.final_w_fusion = self.short.final_w_fusion
        settings_local.final_w_ce = self.short.final_w_ce
        settings_local.final_fusion_norm = self.short.final_fusion_norm
        settings_local.final_ce_score = self.short.final_ce_score

        switches_local.use_opensearch = self.short.use_opensearch
        return settings_local, switches_local

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

        queued_at = await self.redis.hash_get(f"{self.queue.tprefix}{ticket_id}", "queued_at_ms")

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

        settings_local = deepcopy(self.settings)
        switches_local = deepcopy(self.switches)
        short_mode_applied = False

        raw_query = str(pack["query"]).strip()
        # Внешний контракт pack использует array_filters; внутри сохраняем
        # существующий термин token_filters как нормализованное представление.
        need_ack = True
        try:
            raw_array_filters = self._extract_array_filters(pack)
            raw_exact_filters = self._extract_exact_filters(pack)
        except ValueError as validation_error:
            error_message = str(validation_error)
            await self.queue.set_failed(ticket_id, error_message)
            await self.queue.ack(ticket_id)
            need_ack = False
            return {"status": "error", "error": error_message}

        raw_filters: dict[str, list[str] | None] = {
            raw_field: raw_array_filters.get(raw_field) for raw_field in self.token_filter_config.raw_fields
        }
        # Нормализация token-фильтров в терминах token-полей (role_tokens и т.п.).
        token_filters = normalize_request_token_filters(raw_filters, config=self.token_filter_config)
        self.logger.debug(f"Normalized token filters: {token_filters.by_token_field}")

        exact_filters = normalize_request_exact_filters(
            raw_exact_filters,
            config=self.exact_filter_config,
        )
        self.logger.debug(f"Normalized exact filters: {exact_filters.by_filter_field}")

        token_filter_cache_key = token_filters.cache_key_part()
        exact_filter_cache_key = exact_filters.cache_key_part()
        filter_cache_key = f"tokens:{token_filter_cache_key};exact:{exact_filter_cache_key}"

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
            self._metrics_logger(label="🕒 Normalize query (short mode)", start_time=sh_start)
            # self.logger.info(f"query: {query_tokens}")

            if len(query_tokens) <= self.short.mode_limit:
                override_start = time.perf_counter()
                self.logger.info("Используем short настройки для запроса")
                short_mode_applied = True

                settings_local, switches_local = self._apply_short_settings(
                    settings_local=settings_local,
                    switches_local=switches_local,
                )
                self._metrics_logger(label="🕒 Override (short mode)", start_time=override_start)

            else:
                self.logger.info("Используем обычные настройки для запроса")

        # ---- Normalize query ----
        if self.normalize_query:
            normalize_start = time.perf_counter()
            query = normalize_query(query=raw_query, morph=self.morph)
            query_hash = hash_query(query)
            metrics["normalize_time"] = self._metrics_logger("🕒 Normalize query", normalize_start)
        else:
            query = raw_query
            query_hash = raw_query

        top_k = self._resolve_top_k_for_fresh_search(pack, settings_local)

        if switches_local.use_opensearch:
            lex_candidate = "OpenSearch"

        else:
            lex_candidate = ""
        weighted_w_dense = settings_local.w_dense
        weighted_w_lex = settings_local.w_lex if switches_local.use_opensearch else 0.0
        weighted_w_ce = float(getattr(settings_local, "final_w_ce", 0.0))
        rrf_w_dense = settings_local.rrf_w_dense
        rrf_w_lex = settings_local.rrf_w_lex if switches_local.use_opensearch else 0.0
        final_rank_mode = str(getattr(settings_local, "final_rank_mode", "ce_blend"))
        fusion_mode = settings_local.fusion_mode
        if fusion_mode == "rrf":
            dense_search_enabled = settings_local.dense_top_k > 0 and rrf_w_dense > 0
            lexical_search_enabled = (
                bool(switches_local.use_opensearch)
                and settings_local.lex_top_k > 0
                and rrf_w_lex > 0
            )
        else:
            dense_search_enabled = settings_local.dense_top_k > 0 and weighted_w_dense > 0
            lexical_search_enabled = (
                bool(switches_local.use_opensearch)
                and settings_local.lex_top_k > 0
                and weighted_w_lex > 0
            )
        lex_enable = lexical_search_enabled
        cross_encoder_enabled = False
        dense, lex = [], []
        merged: list[dict[str, tp.Any]] | None = None

        try:
            async with self.sem.acquire(), self.uow:
                query_start_time = datetime.now(UTC)
                metrics["semaphore_acquire_time"] = self._metrics_logger("🕒 Semaphore acquire", time.perf_counter())

                search_use_cache = bool(pack.get("search_use_cache", True))
                show_intermediate_results = bool(pack.get("show_intermediate_results", False))
                metrics_enable = bool(pack.get("metrics_enable", False))
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
                hybrid_version = self._build_hybrid_version(
                    settings_local=settings_local,
                    use_opensearch=bool(switches_local.use_opensearch),
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
                allow_cache_read = search_use_cache
                cached = await self.redis.get(cache_key) if allow_cache_read else None
                intermediate_results: dict[str, list[dict[str, tp.Any]]] | None = None
                if not search_use_cache:
                    self.logger.debug("Cache read skipped because request search_use_cache=False")

                if cached:
                    self.logger.info("📦 Выдаем результат из кеша")
                    top_k = self._resolve_top_k_for_cached_response(pack, settings_local)
                    cache_parse_start = time.perf_counter()
                    try:
                        cached_results, cached_intermediate_results = self._load_search_cache_payload(cached)
                    except (json.JSONDecodeError, ValueError) as cache_error:
                        self.logger.warning(f"Search-cache payload parse failed for key={cache_key}: {cache_error}")
                        cached_results = []
                        cached_intermediate_results = None
                    if cached_results:
                        results = cached_results
                        if show_intermediate_results and cached_intermediate_results is not None:
                            intermediate_results = tp.cast(
                                dict[str, list[dict[str, tp.Any]]],
                                cached_intermediate_results,
                            )
                        else:
                            intermediate_results = None
                    else:
                        cached = None
                    metrics["cache_parse_time"] = self._metrics_logger("🕒 Cache parse", cache_parse_start)
                    merged = None
                if not cached:
                    # ---- Presearch ----
                    presearch_result = None
                    if presearch_enabled and presearch_field:
                        presearch_start = time.perf_counter()
                        presearch_use_ce = self._should_run_cross_encoder(
                            items=[{}],
                            final_rank_mode=final_rank_mode,
                            final_w_ce=weighted_w_ce,
                        )
                        presearch_result = await self._presearch_exact_match(
                            query=raw_query,
                            field_name=presearch_field,
                            use_ce=presearch_use_ce,
                        )
                        metrics["presearch_time"] = self._metrics_logger("🕒 Presearch (exact match)", presearch_start)

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
                    metrics["embedding_time"] = self._metrics_logger("🕒 Model encode", encode_start)

                    # ---- Dense & Lex search ----
                    async def _dense_task() -> list[dict[str, tp.Any]]:
                        if not dense_search_enabled:
                            return []
                        start = time.perf_counter()
                        res = await self.vector_db.search(
                            collection_name=settings_local.collection_name,
                            query_vector=query_vector,
                            top_k=settings_local.dense_top_k,
                            filter_expr=milvus_filter_expr,
                        )
                        for d in res:
                            raw_dense = float(d.get("score_dense", 0.0))
                            d["score_dense_raw"] = raw_dense
                            d["score_dense"] = self._dense_to_unit(raw_dense)
                        res = self._precut_dense(res)
                        metrics["vector_search_time"] = self._metrics_logger("🕒 Milvus search", start)
                        return res

                    async def _lex_task() -> list[dict[str, tp.Any]]:
                        if not lexical_search_enabled:
                            return []
                        start = time.perf_counter()
                        res = await self._os_candidates(
                            query=query,
                            k=settings_local.lex_top_k,
                            token_filters=token_filters,
                            exact_filters=exact_filters,
                        )
                        res = self._precut_lex(res)
                        metrics["lexical_search_time"] = self._metrics_logger("🕒 Lexical search", start)
                        return res

                    dense, lex = await asyncio.gather(_dense_task(), _lex_task())

                    if settings_local.fusion_mode == "rrf":
                        merged = self._merge_candidates_rrf(
                            dense=dense,
                            lex=lex,
                            w_dense=rrf_w_dense,
                            w_lex=rrf_w_lex,
                            rrf_k=settings_local.rrf_k,
                        )
                    else:
                        merged = self._merge_candidates(dense, lex)
                        for item in merged:
                            item["score_fusion"] = weighted_w_dense * float(
                                item.get("score_dense", 0.0)
                            ) + weighted_w_lex * float(item.get("score_lex", 0.0))

                    should_run_ce = self._should_run_cross_encoder(
                        items=merged,
                        final_rank_mode=final_rank_mode,
                        final_w_ce=weighted_w_ce,
                    )
                    cross_encoder_enabled = should_run_ce
                    # ---- Cross-encoder stage ----
                    if should_run_ce and merged:
                        start = time.perf_counter()
                        pairs = [(query, self._concat_text(m)) for m in merged]

                        start_rank_fast = time.perf_counter()
                        raw_scores = await asyncio.to_thread(self.ce.rank_fast, pairs)
                        self._metrics_logger(label="🕒 Encoder (rank_fast)", start_time=start_rank_fast)

                        start_postprocess = time.perf_counter()
                        scores = self.ce.ce_postprocess(raw_scores)
                        self._metrics_logger(
                            label="🕒 Encoder (postprocess)",
                            start_time=start_postprocess,
                        )

                        metrics["cross_encoder_time"] = self._metrics_logger("🕒 Cross-encoder rank", start)
                        for m, raw, s in zip(merged, raw_scores, scores, strict=True):
                            m["score_ce_raw"] = float(raw)
                            m["score_ce"] = float(s)
                    elif merged:
                        for m in merged:
                            m.setdefault("score_ce_raw", 0.0)
                            m.setdefault("score_ce", 0.0)
                    else:
                        cross_encoder_enabled = False

                    _results = self._score_and_slice(
                        merged,
                        top_k,
                        use_ce=should_run_ce,
                        w_dense=weighted_w_dense,
                        w_lex=weighted_w_lex,
                        final_w_fusion=float(getattr(settings_local, "final_w_fusion", 1.0)),
                        final_w_ce=weighted_w_ce,
                        final_fusion_norm=str(getattr(settings_local, "final_fusion_norm", "max")),
                        final_ce_score=str(getattr(settings_local, "final_ce_score", "processed")),
                        fusion_mode=fusion_mode,
                        final_rank_mode=final_rank_mode,
                    )
                    if presearch_result:
                        _results = self._inject_presearch_result(
                            results=_results,
                            presearch_result=presearch_result,
                            top_k=top_k,
                            merge_by_field=settings_local.merge_by_field,
                        )

                    results = _results
                    if show_intermediate_results:
                        intermediate_results = {
                            "dense": sorted(
                                dense,
                                key=lambda x: x.get("score_dense", 0.0),
                                reverse=True,
                            )[: self.intermediate_results_top_k],
                            "lex": sorted(lex, key=lambda x: x.get("score_lex", 0.0), reverse=True)[
                                : self.intermediate_results_top_k
                            ],
                            "fusion": sorted(
                                merged or [],
                                key=lambda x: x.get("score_fusion", 0.0),
                                reverse=True,
                            )[: self.intermediate_results_top_k],
                            "ce": (
                                sorted(
                                    merged or [],
                                    key=lambda x: x.get("score_ce_raw", x.get("score_ce", 0.0)),
                                    reverse=True,
                                )[: self.intermediate_results_top_k]
                                if should_run_ce
                                else []
                            ),
                        }

                    cache_payload_raw = self._dump_search_cache_payload(
                        results=results,
                        intermediate_results=intermediate_results,
                    )
                    await self.redis.set(
                        cache_key,
                        cache_payload_raw,
                        ttl=settings_local.cache_ttl,
                    )

                metrics["total_search_time"] = self._metrics_logger("🕒 Total search", start_total)

                payload: dict[str, tp.Any] = {"results": results}
                if show_intermediate_results and intermediate_results is not None:
                    payload["intermediate_results"] = intermediate_results
                metrics["full_search_task_time"] = _now_ms() - int(queued_at)
                search_request: SearchRequestSchema = await self.uow.search_request.create(
                    create_dto=SearchRequestCreateDTO(
                        id=uuid.uuid4(),
                        query=pack["query"],
                        search_start_time=query_start_time,
                        full_execution_time=_convert_to_ms_or_return_0(metrics.get("full_search_task_time")),
                        search_execution_time=_convert_to_ms_or_return_0(metrics.get("total_search_time")),
                        dense_search_time=_convert_to_ms_or_return_0(metrics.get("vector_search_time")),
                        lex_search_time=_convert_to_ms_or_return_0(metrics.get("lexical_search_time")),
                        query_norm_time=_convert_to_ms_or_return_0(metrics.get("normalize_time")),
                        reranker_time=_convert_to_ms_or_return_0(metrics.get("cross_encoder_time")),
                        model_name=self.model_name,
                        reranker_name=self.ce_model_name,
                        reranker_enable=cross_encoder_enabled,
                        lex_enable=lex_enable,
                        from_cache=bool(metrics.get("cache_parse_time")),
                        lex_candidate=lex_candidate,
                        dense_top_k=len(dense),
                        lex_top_k=len(lex),
                        top_k=top_k,
                        weight_dense=settings_local.w_dense,
                        weight_lex=settings_local.w_lex,
                        results=results,
                    )
                )
                payload["search_request_id"] = str(search_request.id)

                score_final_mode = str(final_rank_mode)

                if metrics_enable:
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
                        "dense_search_enabled": dense_search_enabled,
                        "lexical_search_enabled": lexical_search_enabled,
                        "cross_encoder_enabled": cross_encoder_enabled,
                        "reranker_enabled": cross_encoder_enabled,
                        "open_search_enabled": lexical_search_enabled,
                        "from_cache": metrics.get("cache_parse_time", False),
                        "hybrid_dense_top_k": len(dense),
                        "hybrid_lex_top_k": len(lex),
                        "hybrid_top_k": top_k,
                        "hybrid_w_dense": weighted_w_dense,
                        "hybrid_w_lex": weighted_w_lex,
                        # legacy alias for benchmark compatibility.
                        "hybrid_w_ce": weighted_w_ce,
                        "hybrid_rrf_w_dense": rrf_w_dense,
                        "hybrid_rrf_w_lex": rrf_w_lex,
                        "hybrid_fusion_mode": fusion_mode,
                        "hybrid_rrf_k": settings_local.rrf_k,
                        "hybrid_final_rank_mode": final_rank_mode,
                        "hybrid_final_w_fusion": float(getattr(settings_local, "final_w_fusion", 1.0)),
                        "hybrid_final_w_ce": weighted_w_ce,
                        "hybrid_final_fusion_norm": str(getattr(settings_local, "final_fusion_norm", "max")),
                        "hybrid_final_ce_score": str(getattr(settings_local, "final_ce_score", "processed")),
                        "short_mode_applied": short_mode_applied,
                        "hybrid_score_final_mode": score_final_mode,
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
                metrics["queue_ack_time"] = self._metrics_logger("🕒 Queue ACK", ack_start)

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
            {"term": {f"{field_name}.keyword": {"value": query, "case_insensitive": True}}},
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
        fallback_body = {"query": {"bool": {"must": {"match_phrase": {field_name: {"query": query}}}}}}
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
            item["score_ce_raw"] = 0.0

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
            if not (result_key_value and item.get(merge_by_field) and item.get(merge_by_field) == result_key_value)
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

    def _normalize_scores(self, scores: list[float], mode: str) -> list[float]:
        """Нормализует список score по выбранной стратегии.

        Используется в final scoring для режимов `ce_blend` и диагностики.
        Поддерживает:
        - max: деление на максимум;
        - minmax: min-max нормализация.
        """
        if not scores:
            return []
        if mode == "minmax":
            min_score = min(scores)
            max_score = max(scores)
            if max_score == min_score:
                return [1.0 for _ in scores]
            return [(float(score) - min_score) / (max_score - min_score) for score in scores]

        max_score = max(scores)
        if max_score <= 0.0:
            return [0.0 for _ in scores]
        return [float(score) / max_score for score in scores]

    def _score_and_slice(
        self,
        items: list[dict[str, tp.Any]],
        top_k: int,
        *,
        use_ce: bool,
        w_dense: float,
        w_lex: float,
        final_w_fusion: float,
        final_w_ce: float,
        final_fusion_norm: str,
        final_ce_score: str,
        fusion_mode: str,
        final_rank_mode: str,
    ) -> list[dict[str, tp.Any]]:
        """Формирует финальные score согласно HYBRID_FINAL_RANK_MODE и сортирует выдачу."""
        if not items:
            return []

        for it in items:
            it["score_ce_raw"] = float(it.get("score_ce_raw", 0.0))
            it["score_ce"] = float(it.get("score_ce", 0.0))
            it["score_fusion"] = float(it.get("score_fusion", 0.0))

        if final_rank_mode == "legacy_weighted":
            if fusion_mode != "weighted_score":
                self.logger.warning("legacy_weighted is supported only for weighted_score; fallback to fusion_only")
                final_rank_mode = "fusion_only"
            else:
                max_dense = max(float(i.get("score_dense", 0.0)) for i in items) or 1.0
                for it in items:
                    score_dense_norm = float(it.get("score_dense", 0.0)) / max_dense if max_dense else 0.0
                    ce_contrib = float(it.get("score_ce", 0.0)) if use_ce and final_w_ce > 0.0 else 0.0
                    it["score_final"] = w_dense * score_dense_norm + w_lex * float(it.get("score_lex", 0.0)) + final_w_ce * ce_contrib
                    it["score_final_mode"] = "weighted_score_legacy_weighted"
                items.sort(key=lambda x: float(x.get("score_final", 0.0)), reverse=True)
                return items[:top_k]

        if final_rank_mode == "fusion_only":
            for it in items:
                it["score_final"] = float(it.get("score_fusion", 0.0))
                it["score_final_mode"] = f"{fusion_mode}_fusion_only"
            items.sort(
                key=lambda x: (float(x.get("score_final", 0.0)), float(x.get("score_dense", 0.0)), float(x.get("score_lex", 0.0))),
                reverse=True,
            )
            return items[:top_k]

        if final_rank_mode == "ce_final":
            for it in items:
                it["score_final"] = float(it.get("score_ce_raw", 0.0))
                it["score_final_mode"] = f"{fusion_mode}_ce_final"
            items.sort(
                key=lambda x: (float(x.get("score_ce_raw", 0.0)), float(x.get("score_fusion", 0.0))),
                reverse=True,
            )
            return items[:top_k]

        # ce_blend (default)
        fusion_norm_values = self._normalize_scores([float(i.get("score_fusion", 0.0)) for i in items], final_fusion_norm)
        ce_enabled = use_ce and final_w_ce > 0.0
        ce_norm_values: list[float]
        if ce_enabled:
            if final_ce_score == "raw":
                ce_norm_values = self._normalize_scores([float(i.get("score_ce_raw", 0.0)) for i in items], "minmax")
            else:
                ce_norm_values = [float(i.get("score_ce", 0.0)) for i in items]
        else:
            ce_norm_values = [0.0 for _ in items]

        total = final_w_fusion + final_w_ce
        if total > 0.0:
            wf = final_w_fusion / total
            wc = final_w_ce / total
        else:
            wf, wc = 1.0, 0.0

        for it, fusion_norm, ce_norm in zip(items, fusion_norm_values, ce_norm_values, strict=True):
            it["score_fusion_norm"] = float(fusion_norm)
            it["score_ce_norm"] = float(ce_norm)
            it["score_final"] = wf * float(fusion_norm) + wc * float(ce_norm)
            it["score_final_mode"] = f"{fusion_mode}_ce_blend"

        items.sort(
            key=lambda x: (float(x.get("score_final", 0.0)), float(x.get("score_fusion_norm", 0.0)), float(x.get("score_ce_norm", 0.0))),
            reverse=True,
        )
        return items[:top_k]

    def _should_run_cross_encoder(
        self,
        *,
        items: list[dict[str, tp.Any]] | None,
        final_rank_mode: str,
        final_w_ce: float,
    ) -> bool:
        """Определяет, запускать ли cross-encoder для текущей конфигурации поиска."""
        if not items:
            return False
        if final_rank_mode == "ce_final":
            return True
        if final_rank_mode == "ce_blend":
            return final_w_ce > 0.0
        if final_rank_mode == "legacy_weighted":
            return final_w_ce > 0.0
        return False

    def _build_hybrid_version(
        self,
        settings_local: tp.Any,
        *,
        use_opensearch: bool,
    ) -> str:
        """Формирует версию гибридного пайплайна для cache key.

        В строку версии добавляются параметры, которые влияют на итоговую
        выдачу и предотвращают коллизии кеша между режимами с/без reranker.
        """
        return (
            f"{settings_local.version}"
            f":fusion={settings_local.fusion_mode}"
            f":rrf_k={settings_local.rrf_k}"
            f":rrf_w_dense={settings_local.rrf_w_dense}"
            f":rrf_w_lex={settings_local.rrf_w_lex}"
            f":w_dense={settings_local.w_dense}"
            f":w_lex={settings_local.w_lex}"
            f":use_opensearch={int(use_opensearch)}"
            f":final_rank_mode={getattr(settings_local, 'final_rank_mode', 'ce_blend')}"
            f":final_w_fusion={getattr(settings_local, 'final_w_fusion', 1.0)}"
            f":final_w_ce={getattr(settings_local, 'final_w_ce', 0.0)}"
            f":final_fusion_norm={getattr(settings_local, 'final_fusion_norm', 'max')}"
            f":final_ce_score={getattr(settings_local, 'final_ce_score', 'processed')}"
        )

    def _concat_text(self, item: dict[str, tp.Any]) -> str:
        """Собирает текст документа для второй части пары cross-encoder/reranker.

        Поддерживает два режима конфигурации `RERANKER_PAIRS_FIELDS`:
        - список полей (legacy CSV/list): строки склеиваются только из значений полей;
        - словарь field->label: к значению добавляется префикс `label: `.

        Пустые/None-значения пропускаются в обоих режимах. Сложные типы
        сериализуются в JSON, чтобы получить читабельный текст без Python repr.
        """

        def stringify(value: tp.Any) -> str:
            """Безопасно преобразует значение поля кандидата в строку для реранкера."""
            if value is None:
                return ""
            if isinstance(value, (dict, list, tuple)):
                return json.dumps(value, ensure_ascii=False).strip()
            return str(value).strip()

        fields = self.reranker_pairs_fields
        lines: list[str] = []

        # Новый объектный формат: field_name -> человекочитаемая подпись.
        if isinstance(fields, dict):
            for field_name, field_label in fields.items():
                value = stringify(item.get(field_name))
                if not value:
                    continue
                label = field_label.strip() if field_label.strip() else field_name
                lines.append(f"{label}: {value}")
            return "\n".join(lines)

        # Legacy-формат: список полей без подписей, только значения.
        for field_name in fields:
            value = stringify(item.get(field_name))
            if value:
                lines.append(value)

        return "\n".join(lines)

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
            if self.switches.use_opensearch:
                await self.os_adapter.search({"query": {"match_all": {}}}, 1)

            # 5) Реранкер — одно предсказание
            _ = await asyncio.to_thread(self.ce.rank_fast, [("warmup", "warmup")])
            # 6) normalize
            normalize_query("врач", self.morph)
            return True
        except Exception as e:
            self.logger.error(f"Ошибка прогрева: ({type(e)}): {traceback.format_exc()}")
            return False

    def _metrics_logger(self, label: str, start_time: float, precision: int = 4) -> float:
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

    def _dense_to_unit(self, value: float) -> float:
        """Нормализует dense score в [0, 1] с учетом метрики Milvus."""
        score = float(value or 0.0)
        metric = str(self.dense_metric or "IP").upper()

        if metric in {"IP", "COSINE"}:
            return max(0.0, min(1.0, score))

        if metric in {"L2", "EUCLIDEAN"}:
            return 1.0 / (1.0 + max(0.0, score))

        return score

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
