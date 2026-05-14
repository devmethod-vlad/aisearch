"""Тесты режимов fusion и финальной сортировки в HybridSearchOrchestrator."""

from types import SimpleNamespace

from app.services.hybrid_search_orchestrator import HybridSearchOrchestrator


def _build_orchestrator() -> HybridSearchOrchestrator:
    """Создаёт минимальный экземпляр оркестратора без тяжелых зависимостей."""
    orchestrator = HybridSearchOrchestrator.__new__(HybridSearchOrchestrator)
    orchestrator.settings = SimpleNamespace(merge_by_field="ext_id", merge_fields=["question"], fusion_mode="weighted_score", rrf_k=60)
    orchestrator.dense_metric = "IP"
    orchestrator.logger = SimpleNamespace(warning=lambda *_: None)
    return orchestrator


def test_weighted_fusion_only_uses_fusion_score_only() -> None:
    """Проверяет weighted_score+fusion_only: score_final=score_fusion и CE не влияет."""
    o = _build_orchestrator()
    items = [
        {"ext_id": "a", "score_fusion": 0.9, "score_ce_raw": 999.0, "score_dense": 0.1, "score_lex": 0.1},
        {"ext_id": "b", "score_fusion": 0.3, "score_ce_raw": -999.0, "score_dense": 0.9, "score_lex": 0.9},
    ]
    results = o._score_and_slice(items, 2, use_ce=True, w_dense=0.55, w_lex=0.15, final_w_fusion=0.7, final_w_ce=0.3, final_fusion_norm="max", final_ce_score="raw", fusion_mode="weighted_score", final_rank_mode="fusion_only")
    assert [r["ext_id"] for r in results] == ["a", "b"]
    assert results[0]["score_final"] == results[0]["score_fusion"]
    assert results[1]["score_final"] == results[1]["score_fusion"]


def test_rrf_fusion_only_uses_fusion_score_only() -> None:
    """Проверяет rrf+fusion_only: score_final=score_fusion и CE не влияет."""
    o = _build_orchestrator()
    items = [
        {"ext_id": "a", "score_fusion": 0.2, "score_ce_raw": 999.0},
        {"ext_id": "b", "score_fusion": 0.8, "score_ce_raw": -999.0},
    ]
    results = o._score_and_slice(items, 2, use_ce=True, w_dense=0.0, w_lex=0.0, final_w_fusion=1.0, final_w_ce=0.0, final_fusion_norm="max", final_ce_score="processed", fusion_mode="rrf", final_rank_mode="fusion_only")
    assert [r["ext_id"] for r in results] == ["b", "a"]
    assert all(r["score_final"] == r["score_fusion"] for r in results)


def test_weighted_ce_final_uses_raw_ce_and_fusion_tiebreaker() -> None:
    """Проверяет weighted_score+ce_final: score_final=raw CE, tie-breaker по fusion."""
    o = _build_orchestrator()
    items = [
        {"ext_id": "a", "score_fusion": 0.4, "score_ce_raw": 0.5},
        {"ext_id": "b", "score_fusion": 0.9, "score_ce_raw": 0.5},
    ]
    results = o._score_and_slice(items, 2, use_ce=True, w_dense=0.6, w_lex=0.4, final_w_fusion=0.7, final_w_ce=0.3, final_fusion_norm="max", final_ce_score="raw", fusion_mode="weighted_score", final_rank_mode="ce_final")
    assert [r["ext_id"] for r in results] == ["b", "a"]
    assert all(r["score_final"] == r["score_ce_raw"] for r in results)


def test_rrf_ce_final_uses_raw_ce_and_fusion_tiebreaker() -> None:
    """Проверяет rrf+ce_final: score_final=raw CE, tie-breaker по fusion."""
    o = _build_orchestrator()
    items = [
        {"ext_id": "a", "score_fusion": 0.2, "score_ce_raw": 0.6},
        {"ext_id": "b", "score_fusion": 0.9, "score_ce_raw": 0.6},
    ]
    results = o._score_and_slice(items, 2, use_ce=True, w_dense=0.0, w_lex=0.0, final_w_fusion=1.0, final_w_ce=0.0, final_fusion_norm="max", final_ce_score="processed", fusion_mode="rrf", final_rank_mode="ce_final")
    assert [r["ext_id"] for r in results] == ["b", "a"]
    assert all(r["score_final"] == r["score_ce_raw"] for r in results)


def test_weighted_ce_blend_computes_norms_and_weighted_sum() -> None:
    """Проверяет weighted_score+ce_blend: нормализации и итоговые веса."""
    o = _build_orchestrator()
    items = [{"ext_id": "a", "score_fusion": 2.0, "score_ce": 0.8, "score_ce_raw": 10.0}, {"ext_id": "b", "score_fusion": 1.0, "score_ce": 0.2, "score_ce_raw": 0.0}]
    results = o._score_and_slice(items, 2, use_ce=True, w_dense=0.0, w_lex=0.0, final_w_fusion=0.75, final_w_ce=0.25, final_fusion_norm="max", final_ce_score="processed", fusion_mode="weighted_score", final_rank_mode="ce_blend")
    assert results[0]["score_fusion_norm"] == 1.0
    assert results[1]["score_fusion_norm"] == 0.5
    assert results[0]["score_ce_norm"] == 0.8
    assert results[1]["score_ce_norm"] == 0.2
    assert results[0]["score_final"] == 0.95


def test_ce_blend_with_zero_ce_weight_disables_ce_contrib() -> None:
    """Проверяет нулевой вклад CE при final_w_ce=0."""
    o = _build_orchestrator()
    items = [{"ext_id": "a", "score_fusion": 2.0, "score_ce": 0.0}, {"ext_id": "b", "score_fusion": 1.0, "score_ce": 1.0}]
    results = o._score_and_slice(items, 2, use_ce=True, w_dense=0.0, w_lex=0.0, final_w_fusion=1.0, final_w_ce=0.0, final_fusion_norm="max", final_ce_score="processed", fusion_mode="weighted_score", final_rank_mode="ce_blend")
    assert [r["ext_id"] for r in results] == ["a", "b"]
    assert all(r["score_ce_norm"] == 0.0 for r in results)


def test_rrf_ce_blend_weights_normalized_and_ce_source_variants() -> None:
    """Проверяет rrf+ce_blend: нормализацию весов и выбор источника CE score."""
    o = _build_orchestrator()
    base = [{"ext_id": "a", "score_fusion": 2.0, "score_ce": 0.1, "score_ce_raw": 10.0}, {"ext_id": "b", "score_fusion": 1.0, "score_ce": 0.9, "score_ce_raw": 20.0}]
    processed = o._score_and_slice([dict(x) for x in base], 2, use_ce=True, w_dense=0.0, w_lex=0.0, final_w_fusion=2.0, final_w_ce=1.0, final_fusion_norm="max", final_ce_score="processed", fusion_mode="rrf", final_rank_mode="ce_blend")
    raw = o._score_and_slice([dict(x) for x in base], 2, use_ce=True, w_dense=0.0, w_lex=0.0, final_w_fusion=2.0, final_w_ce=1.0, final_fusion_norm="max", final_ce_score="raw", fusion_mode="rrf", final_rank_mode="ce_blend")
    assert processed[0]["score_final"] != raw[0]["score_final"]
    assert raw[0]["score_ce_norm"] == 1.0 and raw[1]["score_ce_norm"] == 0.0


def test_weighted_legacy_mode_formula_and_dense_max_normalization() -> None:
    """Проверяет формулу legacy_weighted и max-нормализацию dense."""
    o = _build_orchestrator()
    items = [{"ext_id": "a", "score_dense": 0.5, "score_lex": 0.2, "score_ce": 1.0, "score_ce_raw": -99.0}, {"ext_id": "b", "score_dense": 0.25, "score_lex": 0.2, "score_ce": 0.0, "score_ce_raw": 99.0}]
    results = o._score_and_slice(items, 2, use_ce=True, w_dense=1.0, w_lex=0.0, final_w_fusion=1.0, final_w_ce=0.5, final_fusion_norm="max", final_ce_score="raw", fusion_mode="weighted_score", final_rank_mode="legacy_weighted")
    assert results[0]["score_final"] == 1.5
    assert results[1]["score_final"] == 0.5


def test_should_run_cross_encoder_modes() -> None:
    """Проверяет запуск/пропуск CE по final_rank_mode и final_w_ce."""
    o = _build_orchestrator()
    items = [{"ext_id": "a"}]
    assert not o._should_run_cross_encoder(items=items, final_rank_mode="fusion_only", final_w_ce=1.0)
    assert o._should_run_cross_encoder(items=items, final_rank_mode="ce_final", final_w_ce=0.0)
    assert o._should_run_cross_encoder(items=items, final_rank_mode="ce_blend", final_w_ce=0.1)
    assert not o._should_run_cross_encoder(items=items, final_rank_mode="ce_blend", final_w_ce=0.0)
    assert o._should_run_cross_encoder(items=items, final_rank_mode="legacy_weighted", final_w_ce=0.1)
    assert not o._should_run_cross_encoder(items=items, final_rank_mode="legacy_weighted", final_w_ce=0.0)


def test_build_hybrid_version_contains_new_final_settings_only() -> None:
    """Проверяет cache key по новым final параметрам и отсутствие legacy-флагов."""
    o = _build_orchestrator()
    base = dict(version="v2", fusion_mode="rrf", rrf_k=42, rrf_w_dense=1.2, rrf_w_lex=0.9, w_dense=0.5, w_lex=0.5)
    v1 = o._build_hybrid_version(SimpleNamespace(**base, final_rank_mode="fusion_only", final_w_fusion=1.0, final_w_ce=0.0, final_fusion_norm="max", final_ce_score="processed"), use_opensearch=True)
    v2 = o._build_hybrid_version(SimpleNamespace(**base, final_rank_mode="ce_final", final_w_fusion=1.0, final_w_ce=0.0, final_fusion_norm="max", final_ce_score="processed"), use_opensearch=True)
    v3 = o._build_hybrid_version(SimpleNamespace(**base, final_rank_mode="ce_blend", final_w_fusion=0.8, final_w_ce=0.2, final_fusion_norm="max", final_ce_score="processed"), use_opensearch=True)
    v4 = o._build_hybrid_version(SimpleNamespace(**base, final_rank_mode="legacy_weighted", final_w_fusion=0.8, final_w_ce=0.2, final_fusion_norm="minmax", final_ce_score="raw"), use_opensearch=True)
    assert v1 != v2 and v3 != v4
    assert ":final_w_ce=0.2" in v3 and ":final_w_fusion=0.8" in v3
    assert ":final_fusion_norm=minmax" in v4 and ":final_ce_score=raw" in v4
    assert ":final_rank_mode=fusion_only" in v1 and ":final_rank_mode=ce_final" in v2


def test_apply_short_settings_overrides_final_and_switches() -> None:
    """Проверяет short override для fusion/final/switches, включая use_opensearch."""
    o = _build_orchestrator()
    o.short = SimpleNamespace(top_k=3, w_lex=0.7, w_dense=0.3, final_w_fusion=0.6, final_w_ce=0.3, final_fusion_norm="minmax", final_ce_score="raw", rrf_w_dense=2.0, rrf_w_lex=3.0, dense_top_k=11, lex_top_k=12, fusion_mode="rrf", rrf_k=30, use_opensearch=True, mode=True, mode_limit=4)
    settings_local = SimpleNamespace(top_k=10, w_lex=0.15, w_dense=0.55, final_w_fusion=0.8, final_w_ce=0.1, final_fusion_norm="max", final_ce_score="processed", rrf_w_dense=1.0, rrf_w_lex=1.0, dense_top_k=50, lex_top_k=20, fusion_mode="weighted_score", rrf_k=60)
    switches_local = SimpleNamespace(use_opensearch=False)
    updated_settings, updated_switches = o._apply_short_settings(settings_local=settings_local, switches_local=switches_local)
    assert updated_settings.fusion_mode == "rrf"
    assert updated_settings.final_w_ce == 0.3
    assert updated_settings.final_w_fusion == 0.6
    assert updated_settings.final_fusion_norm == "minmax"
    assert updated_settings.final_ce_score == "raw"
    assert updated_switches.use_opensearch is True
