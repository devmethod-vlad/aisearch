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
    orchestrator.logger = SimpleNamespace(warning=lambda *_: None)
    return orchestrator


def test_weighted_fusion_without_ce() -> None:
    """Проверяет weighted fusion без CE: score_final должен совпадать со score_fusion."""
    orchestrator = _build_orchestrator()
    merged = [{"ext_id": "a", "score_dense": 0.8, "score_lex": 0.2}]

    results = orchestrator._score_and_slice(
        merged,
        1,
        use_ce=False,
        w_dense=0.55,
        w_lex=0.15,
        fusion_mode="weighted_score",
    )

    expected = 0.55 * 0.8 + 0.15 * 0.2
    assert results[0]["score_fusion"] == expected
    assert results[0]["score_final"] == expected


def test_weighted_fusion_with_ce_final_sorting() -> None:
    """Проверяет, что CE определяет финальный порядок, а score_fusion работает tie-breaker'ом."""
    orchestrator = _build_orchestrator()
    merged = [
        {"ext_id": "a", "score_dense": 0.9, "score_lex": 0.7, "score_ce": 0.2},
        {"ext_id": "b", "score_dense": 0.1, "score_lex": 0.2, "score_ce": 0.9},
        {"ext_id": "c", "score_dense": 0.2, "score_lex": 0.1, "score_ce": 0.9},
    ]

    results = orchestrator._score_and_slice(
        merged,
        3,
        use_ce=True,
        w_dense=0.55,
        w_lex=0.15,
        fusion_mode="weighted_score",
    )

    assert results[0]["ext_id"] == "c"
    assert results[1]["ext_id"] == "b"
    assert results[2]["ext_id"] == "a"
    assert results[0]["score_final"] == results[0]["score_ce"]


def test_merge_candidates_rrf() -> None:
    """Проверяет RRF merge, нормализацию score_fusion и объединение источников."""
    orchestrator = _build_orchestrator()
    dense = [
        {"ext_id": "a", "question": "A", "score_dense": 0.9},
        {"ext_id": "b", "question": "B", "score_dense": 0.8},
    ]
    lex = [
        {"ext_id": "b", "question": "B", "score_lex": 1.0, "_source": "opensearch"},
        {"ext_id": "c", "question": "C", "score_lex": 0.7, "_source": "opensearch"},
    ]

    merged = orchestrator._merge_candidates_rrf(
        dense, lex, w_dense=0.55, w_lex=0.15, rrf_k=60
    )
    by_id = {item["ext_id"]: item for item in merged}

    assert {"a", "b", "c"} == set(by_id)
    assert "dense" in by_id["b"]["sources"]
    assert "opensearch" in by_id["b"]["sources"]
    assert all("score_rrf_raw" in item for item in merged)
    assert all("score_fusion" in item for item in merged)
    assert all(0.0 <= item["score_fusion"] <= 1.0 for item in merged)
    assert by_id["b"]["score_rrf_raw"] >= by_id["a"]["score_rrf_raw"]


def test_rrf_without_ce_uses_fusion_for_final_sort() -> None:
    """Проверяет, что при RRF без CE финальная сортировка идет по score_fusion."""
    orchestrator = _build_orchestrator()
    merged = [
        {"ext_id": "a", "score_fusion": 0.9},
        {"ext_id": "b", "score_fusion": 0.5},
    ]

    results = orchestrator._score_and_slice(
        merged,
        2,
        use_ce=False,
        w_dense=0.55,
        w_lex=0.15,
        fusion_mode="rrf",
    )

    assert results[0]["ext_id"] == "a"
    assert results[0]["score_final"] == results[0]["score_fusion"]


def test_rrf_with_ce_final_sorting() -> None:
    """Проверяет приоритет CE над fusion в RRF-режиме и tie-breaker по score_fusion."""
    orchestrator = _build_orchestrator()
    merged = [
        {"ext_id": "a", "score_fusion": 0.95, "score_ce": 0.1},
        {"ext_id": "b", "score_fusion": 0.25, "score_ce": 0.9},
        {"ext_id": "c", "score_fusion": 0.5, "score_ce": 0.9},
    ]

    results = orchestrator._score_and_slice(
        merged,
        3,
        use_ce=True,
        w_dense=0.55,
        w_lex=0.15,
        fusion_mode="rrf",
    )

    assert results[0]["ext_id"] == "c"
    assert results[1]["ext_id"] == "b"
    assert results[2]["ext_id"] == "a"
    assert results[0]["score_final"] == results[0]["score_ce"]


def test_build_hybrid_version_contains_reranker_flag() -> None:
    """Проверяет, что версия пайплайна учитывает effective-флаг reranker."""
    orchestrator = _build_orchestrator()
    settings = SimpleNamespace(version="v2", fusion_mode="rrf", rrf_k=42)

    enabled_version = orchestrator._build_hybrid_version(
        settings_local=settings,
        reranker_enabled=True,
    )
    disabled_version = orchestrator._build_hybrid_version(
        settings_local=settings,
        reranker_enabled=False,
    )

    assert ":reranker=1" in enabled_version
    assert ":reranker=0" in disabled_version
    assert enabled_version != disabled_version


def test_apply_short_settings_overrides_fusion_and_rrf_k() -> None:
    """Проверяет, что short override подменяет fusion_mode и rrf_k в effective settings."""
    orchestrator = _build_orchestrator()
    orchestrator.short = SimpleNamespace(
        top_k=3,
        w_lex=0.7,
        w_dense=0.3,
        dense_top_k=11,
        lex_top_k=12,
        fusion_mode="rrf",
        rrf_k=30,
        use_hybrid=True,
        use_opensearch=True,
        use_reranker=False,
        mode=True,
        mode_limit=4,
    )
    settings_local = SimpleNamespace(
        top_k=10,
        w_lex=0.15,
        w_dense=0.55,
        dense_top_k=50,
        lex_top_k=20,
        fusion_mode="weighted_score",
        rrf_k=60,
    )
    switches_local = SimpleNamespace(
        use_hybrid=False,
        use_opensearch=False,
        use_reranker=True,
    )

    updated_settings, _ = orchestrator._apply_short_settings(
        settings_local=settings_local,
        switches_local=switches_local,
    )

    assert updated_settings.fusion_mode == "rrf"
    assert updated_settings.rrf_k == 30
