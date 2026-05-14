from types import SimpleNamespace

from app.services.hybrid_search_orchestrator import HybridSearchOrchestrator


def _build_orchestrator_with_os_config(**config_overrides: object) -> HybridSearchOrchestrator:
    """Создаёт orchestrator с минимальным stub-конфигом OpenSearch для тестирования builder-методов."""
    orchestrator = HybridSearchOrchestrator.__new__(HybridSearchOrchestrator)
    config = {
        "search_fields": ["question", "analysis"],
        "output_fields": ["ext_id", "question"],
        "operator": "or",
        "min_should_match": "1",
        "multi_match_type": "best_fields",
        "fuzziness": 0,
        "phrase_field_boosts": {},
        "phrase_slop": 0,
        "bool_min_should_match": 1,
    }
    config.update(config_overrides)
    orchestrator.os_adapter = SimpleNamespace(config=SimpleNamespace(**config))
    return orchestrator


def test_build_os_search_body_simple_mode_by_default() -> None:
    """Проверяет simple-режим: bool.must с multi_match без should при пустых phrase boost'ах."""
    orchestrator = _build_orchestrator_with_os_config(phrase_field_boosts={})

    body = orchestrator._build_os_search_body(query="test", filter_clauses=[])

    assert "must" in body["query"]["bool"]
    assert "multi_match" in body["query"]["bool"]["must"]
    assert "should" not in body["query"]["bool"]
    assert body["_source"] == orchestrator.os_adapter.config.output_fields
    assert body["track_total_hits"] is False


def test_build_os_search_body_multi_signal_mode() -> None:
    """Проверяет multi-signal режим: should содержит phrase-сигналы и основной multi_match."""
    orchestrator = _build_orchestrator_with_os_config(
        phrase_field_boosts={"question": 6.0, "answer": 2.0}
    )
    filter_clauses = [{"term": {"role_tokens": "врач"}}]

    body = orchestrator._build_os_search_body(query="test", filter_clauses=filter_clauses)

    should_clauses = body["query"]["bool"]["should"]
    assert "must" not in body["query"]["bool"]
    assert body["query"]["bool"]["filter"] == filter_clauses
    assert body["query"]["bool"]["minimum_should_match"] == 1
    assert len([c for c in should_clauses if "match_phrase" in c]) == 2
    assert len([c for c in should_clauses if "multi_match" in c]) == 1


def test_build_os_phrase_should_clauses_adds_slop_only_when_positive() -> None:
    """Проверяет добавление slop в phrase-сигналы только при OS_PHRASE_SLOP > 0."""
    orchestrator = _build_orchestrator_with_os_config(
        phrase_field_boosts={"question": 6.0},
        phrase_slop=2,
    )
    clauses = orchestrator._build_os_phrase_should_clauses("test")
    assert clauses[0]["match_phrase"]["question"]["slop"] == 2

    orchestrator_no_slop = _build_orchestrator_with_os_config(
        phrase_field_boosts={"question": 6.0},
        phrase_slop=0,
    )
    no_slop_clauses = orchestrator_no_slop._build_os_phrase_should_clauses("test")
    assert "slop" not in no_slop_clauses[0]["match_phrase"]["question"]


def test_build_os_multi_match_query_respects_operator_minimum_should_match_and_type() -> None:
    """Проверяет поведение multi_match: type всегда передаётся, minimum_should_match только для operator=or."""
    orchestrator_or = _build_orchestrator_with_os_config(
        operator="or",
        min_should_match="70%",
        multi_match_type="cross_fields",
    )
    mm_or = orchestrator_or._build_os_multi_match_query("test")["multi_match"]
    assert mm_or["type"] == "cross_fields"
    assert mm_or["minimum_should_match"] == "70%"

    orchestrator_and = _build_orchestrator_with_os_config(
        operator="and",
        min_should_match="70%",
        multi_match_type="cross_fields",
    )
    mm_and = orchestrator_and._build_os_multi_match_query("test")["multi_match"]
    assert mm_and["type"] == "cross_fields"
    assert "minimum_should_match" not in mm_and
