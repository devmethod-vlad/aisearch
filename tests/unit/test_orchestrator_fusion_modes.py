"""Тесты режимов fusion и финальной сортировки в HybridSearchOrchestrator."""

from types import SimpleNamespace

from app.services.hybrid_search_orchestrator import HybridSearchOrchestrator


def _build_orchestrator() -> HybridSearchOrchestrator:
    """Создаёт минимальный экземпляр оркестратора без инициализации тяжелых зависимостей."""
    orchestrator = HybridSearchOrchestrator.__new__(HybridSearchOrchestrator)
    orchestrator.settings = SimpleNamespace(
        merge_by_field="ext_id",
        merge_fields=["question"],
        fusion_mode="weighted_score",
        rrf_k=60,
    )
    orchestrator.dense_metric = "IP"
    orchestrator.logger = SimpleNamespace(warning=lambda *_: None)
    return orchestrator


def test_dense_to_unit_metrics() -> None:
    """Проверяет преобразование dense score для разных метрик."""
    o = _build_orchestrator()
    o.dense_metric = "IP"
    assert o._dense_to_unit(0.8) == 0.8
    assert o._dense_to_unit(1.2) == 1.0
    assert o._dense_to_unit(-0.2) == 0.0


def test_weighted_ce_final_mode_uses_raw_ce_sort() -> None:
    """Проверяет weighted_score+ce_final: сортировка по raw CE, tie-breaker по fusion."""
    o = _build_orchestrator()
    merged = [
        {"ext_id": "a", "score_dense": 0.9, "score_lex": 0.9, "score_ce_raw": 0.1, "score_ce": 0.9},
        {"ext_id": "b", "score_dense": 0.2, "score_lex": 0.2, "score_ce_raw": 0.8, "score_ce": 0.2},
    ]
    results = o._score_and_slice(
        merged,
        2,
        use_ce=True,
        w_dense=0.55,
        w_lex=0.15,
        final_w_fusion=0.6,
        final_w_ce=0.3,
        final_fusion_norm="minmax",
        final_ce_score="raw",
        fusion_mode="weighted_score",
        ce_as_final_rank=True,
    )
    assert results[0]["ext_id"] == "b"
    assert results[0]["score_final"] == results[0]["score_ce_raw"]


def test_weighted_legacy_mode_uses_legacy_formula() -> None:
    """Проверяет legacy weighted formula и сортировку по score_final."""
    o = _build_orchestrator()
    items = [
        {"ext_id": "a", "score_dense": 0.9, "score_lex": 0.1, "score_ce": 0.1, "score_ce_raw": 10.0},
        {"ext_id": "b", "score_dense": 0.6, "score_lex": 0.1, "score_ce": 1.0, "score_ce_raw": 0.1},
    ]
    results = o._score_and_slice(
        items, 2, use_ce=True, w_dense=0.5, w_lex=0.2, w_ce=0.4, fusion_mode="weighted_score", ce_as_final_rank=False
    )
    assert results[0]["ext_id"] == "b"
    assert results[0]["score_final"] > results[1]["score_final"]


def test_weighted_legacy_uses_score_ce_not_raw() -> None:
    """Проверяет, что legacy weighted использует processed score_ce, а не score_ce_raw."""
    o = _build_orchestrator()
    items = [
        {"ext_id": "a", "score_dense": 0.5, "score_lex": 0.5, "score_ce": 0.1, "score_ce_raw": 99.0},
        {"ext_id": "b", "score_dense": 0.5, "score_lex": 0.5, "score_ce": 0.9, "score_ce_raw": -99.0},
    ]
    results = o._score_and_slice(
        items, 2, use_ce=True, w_dense=0.4, w_lex=0.4, w_ce=0.2, fusion_mode="weighted_score", ce_as_final_rank=False
    )
    assert results[0]["ext_id"] == "b"


def test_weighted_legacy_w_ce_zero_disables_ce_contrib() -> None:
    """Проверяет, что при w_ce=0 CE-вклад в legacy mode отключается."""
    o = _build_orchestrator()
    items = [
        {"ext_id": "a", "score_dense": 0.5, "score_lex": 0.0, "score_ce": 1.0},
        {"ext_id": "b", "score_dense": 0.25, "score_lex": 0.0, "score_ce": 0.0},
    ]
    results = o._score_and_slice(
        items, 2, use_ce=True, w_dense=1.0, w_lex=0.0, w_ce=0.0, fusion_mode="weighted_score", ce_as_final_rank=False
    )
    assert results[0]["score_final"] == 1.0
    assert results[1]["score_final"] == 0.5


def test_weighted_legacy_use_ce_false_ignores_ce() -> None:
    """Проверяет, что при use_ce=False CE-вклад игнорируется."""
    o = _build_orchestrator()
    items = [
        {"ext_id": "a", "score_dense": 0.5, "score_lex": 0.0, "score_ce": 0.0},
        {"ext_id": "b", "score_dense": 0.25, "score_lex": 0.0, "score_ce": 1.0},
    ]
    results = o._score_and_slice(
        items, 2, use_ce=False, w_dense=1.0, w_lex=0.0, w_ce=0.9, fusion_mode="weighted_score", ce_as_final_rank=False
    )
    assert results[0]["ext_id"] == "a"


def test_weighted_legacy_dense_max_normalization() -> None:
    """Проверяет нормализацию dense через max_dense в legacy weighted режиме."""
    o = _build_orchestrator()
    items = [
        {"ext_id": "a", "score_dense": 0.5, "score_lex": 0.0, "score_ce": 0.0},
        {"ext_id": "b", "score_dense": 0.25, "score_lex": 0.0, "score_ce": 0.0},
    ]
    results = o._score_and_slice(
        items, 2, use_ce=False, w_dense=1.0, w_lex=0.0, w_ce=0.0, fusion_mode="weighted_score", ce_as_final_rank=False
    )
    assert results[0]["score_final"] == 1.0
    assert results[1]["score_final"] == 0.5


def test_rrf_ignores_ce_as_final_rank_flag() -> None:
    """Проверяет, что RRF остаётся CE-final даже при ce_as_final_rank=False."""
    o = _build_orchestrator()
    merged = [
        {"ext_id": "a", "score_fusion": 0.9, "score_ce_raw": 0.1},
        {"ext_id": "b", "score_fusion": 0.1, "score_ce_raw": 0.9},
    ]
    results = o._score_and_slice(
        merged, 2, use_ce=True, w_dense=0.0, w_lex=0.0, w_ce=0.0, fusion_mode="rrf", ce_as_final_rank=False
    )
    assert results[0]["ext_id"] == "b"


def test_should_run_cross_encoder_matrix() -> None:
    """Проверяет матрицу условий запуска cross-encoder."""
    o = _build_orchestrator()
    items = [{"ext_id": "a"}]
    assert o._should_run_cross_encoder(
        items=items, use_reranker=True, fusion_mode="rrf", ce_as_final_rank=False, w_ce=0.0
    )
    assert not o._should_run_cross_encoder(
        items=items, use_reranker=False, fusion_mode="rrf", ce_as_final_rank=True, w_ce=0.5
    )
    assert o._should_run_cross_encoder(
        items=items, use_reranker=True, fusion_mode="weighted_score", ce_as_final_rank=True, w_ce=0.0
    )
    assert o._should_run_cross_encoder(
        items=items, use_reranker=True, fusion_mode="weighted_score", ce_as_final_rank=False, w_ce=0.1
    )
    assert not o._should_run_cross_encoder(
        items=items, use_reranker=True, fusion_mode="weighted_score", ce_as_final_rank=False, w_ce=0.0
    )
    assert not o._should_run_cross_encoder(
        items=items, use_reranker=False, fusion_mode="weighted_score", ce_as_final_rank=False, w_ce=0.1
    )


def test_build_hybrid_version_contains_ce_flags() -> None:
    """Проверяет включение ce_final и w_ce в cache-version."""
    o = _build_orchestrator()
    settings_on = SimpleNamespace(
        version="v2", fusion_mode="rrf", rrf_k=42, rrf_w_dense=1.2, rrf_w_lex=0.9, ce_as_final_rank=True, w_ce=0.3
    )
    settings_off = SimpleNamespace(
        version="v2", fusion_mode="rrf", rrf_k=42, rrf_w_dense=1.2, rrf_w_lex=0.9, ce_as_final_rank=False, w_ce=0.0
    )
    v_on = o._build_hybrid_version(settings_local=settings_on, reranker_enabled=True)
    v_off = o._build_hybrid_version(settings_local=settings_off, reranker_enabled=True)
    assert ":ce_final=1" in v_on
    assert ":w_ce=0.3" in v_on
    assert ":ce_final=0" in v_off
    assert ":w_ce=0.0" in v_off
    assert v_on != v_off


def test_apply_short_settings_overrides_rrf_weights_and_final_settings() -> None:
    """Проверяет, что short override подменяет rrf-веса и final-настройки."""
    o = _build_orchestrator()
    o.short = SimpleNamespace(
        top_k=3,
        w_lex=0.7,
        w_dense=0.3,
        final_w_fusion=0.6,
        final_w_ce=0.3,
        final_fusion_norm="minmax",
        final_ce_score="raw",
        rrf_w_dense=2.0,
        rrf_w_lex=3.0,
        dense_top_k=11,
        lex_top_k=12,
        fusion_mode="rrf",
        rrf_k=30,
        use_opensearch=True,
        mode=True,
        mode_limit=4,
    )
    settings_local = SimpleNamespace(
        top_k=10,
        w_lex=0.15,
        w_dense=0.55,
        final_w_fusion=0.8,
        final_w_ce=0.1,
        final_fusion_norm="max",
        final_ce_score="processed",
        rrf_w_dense=1.0,
        rrf_w_lex=1.0,
        dense_top_k=50,
        lex_top_k=20,
        fusion_mode="weighted_score",
        rrf_k=60,
    )
    switches_local = SimpleNamespace(use_opensearch=False)
    updated_settings, _ = o._apply_short_settings(settings_local=settings_local, switches_local=switches_local)
    assert updated_settings.final_w_ce == 0.3
    assert updated_settings.final_w_fusion == 0.6
    assert updated_settings.final_fusion_norm == "minmax"
    assert updated_settings.final_ce_score == "raw"
    assert updated_settings.rrf_w_dense == 2.0
    assert updated_settings.rrf_w_lex == 3.0
