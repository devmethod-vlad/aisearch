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
    o.dense_metric = "COSINE"
    assert o._dense_to_unit(0.8) == 0.8
    assert o._dense_to_unit(1.2) == 1.0
    assert o._dense_to_unit(-0.2) == 0.0
    o.dense_metric = "L2"
    assert o._dense_to_unit(0.0) == 1.0
    assert o._dense_to_unit(1.0) == 0.5
    assert o._dense_to_unit(-1.0) == 1.0
    o.dense_metric = "UNKNOWN"
    assert o._dense_to_unit(2) == 2.0


def test_weighted_fusion_without_ce() -> None:
    """Проверяет weighted fusion без CE: score_final должен совпадать со score_fusion."""
    orchestrator = _build_orchestrator()
    merged = [{"ext_id": "a", "score_dense": 0.8, "score_lex": 0.2}]
    results = orchestrator._score_and_slice(merged, 1, use_ce=False, w_dense=0.55, w_lex=0.15, fusion_mode="weighted_score")
    expected = 0.55 * 0.8 + 0.15 * 0.2
    assert results[0]["score_fusion"] == expected
    assert results[0]["score_final"] == expected


def test_ce_sort_uses_raw_then_fusion() -> None:
    """Проверяет CE сортировку по score_ce_raw и tie-breaker по score_fusion."""
    o = _build_orchestrator()
    merged = [
        {"ext_id": "a", "score_fusion": 0.95, "score_ce_raw": 0.1, "score_ce": 0.9},
        {"ext_id": "b", "score_fusion": 0.25, "score_ce_raw": 0.9, "score_ce": 0.2},
        {"ext_id": "c", "score_fusion": 0.5, "score_ce_raw": 0.9, "score_ce": 0.1},
    ]
    results = o._score_and_slice(merged, 3, use_ce=True, w_dense=0.55, w_lex=0.15, fusion_mode="rrf")
    assert [r["ext_id"] for r in results] == ["c", "b", "a"]
    assert results[0]["score_final"] == results[0]["score_ce_raw"]


def test_ce_sort_fallback_to_processed_score() -> None:
    """Проверяет fallback сортировки по score_ce, если score_ce_raw отсутствует."""
    o = _build_orchestrator()
    merged = [
        {"ext_id": "a", "score_fusion": 0.1, "score_ce": 0.2},
        {"ext_id": "b", "score_fusion": 0.9, "score_ce": 0.8},
    ]
    results = o._score_and_slice(merged, 2, use_ce=True, w_dense=0.55, w_lex=0.15, fusion_mode="rrf")
    assert [r["ext_id"] for r in results] == ["b", "a"]


def test_merge_candidates_rrf() -> None:
    """Проверяет RRF merge, нормализацию score_fusion и объединение источников."""
    orchestrator = _build_orchestrator()
    dense = [{"ext_id": "a", "question": "A", "score_dense": 0.9, "score_dense_raw": 0.9}, {"ext_id": "b", "question": "B", "score_dense": 0.8, "score_dense_raw": 0.8}]
    lex = [{"ext_id": "b", "question": "B", "score_lex": 1.0, "score_lex_raw": 10.0, "_source": "opensearch"}, {"ext_id": "c", "question": "C", "score_lex": 0.7, "score_lex_raw": 7.0, "_source": "opensearch"}]
    merged = orchestrator._merge_candidates_rrf(dense, lex, w_dense=2.0, w_lex=1.0, rrf_k=60)
    by_id = {item["ext_id"]: item for item in merged}
    assert by_id["b"]["score_rrf_raw"] >= by_id["a"]["score_rrf_raw"]
    assert "score_dense_raw" in by_id["b"]
    assert "score_lex_raw" in by_id["b"]


def test_build_hybrid_version_contains_reranker_flag_and_rrf_weights() -> None:
    """Проверяет включение reranker-флага и RRF-весов в версию cache-key."""
    o = _build_orchestrator()
    settings = SimpleNamespace(version="v2", fusion_mode="rrf", rrf_k=42, rrf_w_dense=1.2, rrf_w_lex=0.9)
    v = o._build_hybrid_version(settings_local=settings, reranker_enabled=True)
    assert ":reranker=1" in v and ":rrf_w_dense=1.2" in v and ":rrf_w_lex=0.9" in v


def test_apply_short_settings_overrides_rrf_weights() -> None:
    """Проверяет, что short override подменяет rrf-веса."""
    o = _build_orchestrator()
    o.short = SimpleNamespace(top_k=3, w_lex=0.7, w_dense=0.3, rrf_w_dense=2.0, rrf_w_lex=3.0, dense_top_k=11, lex_top_k=12, fusion_mode="rrf", rrf_k=30, use_hybrid=True, use_opensearch=True, use_reranker=False, mode=True, mode_limit=4)
    settings_local = SimpleNamespace(top_k=10, w_lex=0.15, w_dense=0.55, rrf_w_dense=1.0, rrf_w_lex=1.0, dense_top_k=50, lex_top_k=20, fusion_mode="weighted_score", rrf_k=60)
    switches_local = SimpleNamespace(use_hybrid=False, use_opensearch=False, use_reranker=True)
    updated_settings, _ = o._apply_short_settings(settings_local=settings_local, switches_local=switches_local)
    assert updated_settings.rrf_w_dense == 2.0 and updated_settings.rrf_w_lex == 3.0
