"""Тесты парсинга RERANKER_PAIRS_FIELDS и сборки текста для reranker."""

import pytest
from pydantic import ValidationError

from app.settings.config import RerankerSettings
from tests.unit.test_orchestrator_presearch_pipeline import _build_orchestrator


def test_reranker_pairs_fields_csv_format() -> None:
    """Проверяет, что CSV-строка корректно парсится в список полей."""
    settings = RerankerSettings(
        model_name="test-reranker",
        pairs_fields="question,analysis,answer",
    )

    assert settings.pairs_fields == ["question", "analysis", "answer"]


def test_reranker_pairs_fields_json_object_format() -> None:
    """Проверяет, что JSON-объект парсится в mapping field->label."""
    settings = RerankerSettings(
        model_name="test-reranker",
        pairs_fields='{"question":"Вопрос","analysis":"Анализ","answer":"Ответ"}',
    )

    assert settings.pairs_fields == {
        "question": "Вопрос",
        "analysis": "Анализ",
        "answer": "Ответ",
    }


def test_reranker_pairs_fields_python_dict_like_fallback() -> None:
    """Проверяет fallback через ast.literal_eval для dict-подобной строки."""
    settings = RerankerSettings(
        model_name="test-reranker",
        pairs_fields="{'question':'Вопрос','analysis':'Анализ','answer':'Ответ'}",
    )

    assert settings.pairs_fields == {
        "question": "Вопрос",
        "analysis": "Анализ",
        "answer": "Ответ",
    }


def test_concat_text_for_legacy_csv_mode() -> None:
    """Проверяет legacy-режим сборки текста: значения без подписей."""
    orchestrator = _build_orchestrator()
    orchestrator.reranker_pairs_fields = ["question", "analysis", "answer"]

    text = orchestrator._concat_text(
        {
            "question": "Как оформить доступ?",
            "analysis": "Нужно создать заявку.",
            "answer": "Оформите заявку через портал.",
        }
    )

    assert text == (
        "Как оформить доступ?\n"
        "Нужно создать заявку.\n"
        "Оформите заявку через портал."
    )


def test_concat_text_for_object_mode() -> None:
    """Проверяет object-режим: к значениям добавляются label-префиксы."""
    orchestrator = _build_orchestrator()
    orchestrator.reranker_pairs_fields = {
        "question": "Вопрос",
        "analysis": "Анализ",
        "answer": "Ответ",
    }

    text = orchestrator._concat_text(
        {
            "question": "Как оформить доступ?",
            "analysis": "Нужно создать заявку.",
            "answer": "Оформите заявку через портал.",
        }
    )

    assert text == (
        "Вопрос: Как оформить доступ?\n"
        "Анализ: Нужно создать заявку.\n"
        "Ответ: Оформите заявку через портал."
    )


def test_concat_text_skips_empty_values_in_object_mode() -> None:
    """Проверяет пропуск пустых и None-значений при object-конфигурации."""
    orchestrator = _build_orchestrator()
    orchestrator.reranker_pairs_fields = {
        "question": "Вопрос",
        "analysis": "Анализ",
        "answer": "Ответ",
    }

    text = orchestrator._concat_text(
        {
            "question": "Как оформить доступ?",
            "analysis": "",
            "answer": None,
        }
    )

    assert text == "Вопрос: Как оформить доступ?"


def test_reranker_pairs_fields_invalid_object_format() -> None:
    """Проверяет понятную ошибку на некорректном объектном формате."""
    with pytest.raises(ValidationError, match="pairs_fields: некорректный объектный формат"):
        RerankerSettings(
            model_name="test-reranker",
            pairs_fields="{bad json",
        )
