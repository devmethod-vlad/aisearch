import pytest
from pydantic import ValidationError

from app.settings.config import OpenSearchSettings


def _build_settings(**overrides: object) -> OpenSearchSettings:
    """Создаёт валидный OpenSearchSettings для точечных проверок валидации и нормализации."""
    base = {
        "host": "localhost",
        "port": 9200,
        "index_name": "kb",
    }
    base.update(overrides)
    return OpenSearchSettings(**base)


def test_operator_normalizes_or_and() -> None:
    """Проверяет нормализацию OS_OPERATOR к нижнему регистру для OR и AND."""
    assert _build_settings(operator="OR").operator == "or"
    assert _build_settings(operator="AND").operator == "and"


def test_operator_validation_rejects_unknown_value() -> None:
    """Проверяет, что недопустимое значение OS_OPERATOR отклоняется валидацией."""
    with pytest.raises(ValidationError):
        _build_settings(operator="xor")


def test_multi_match_type_normalizes_value() -> None:
    """Проверяет нормализацию OS_MULTI_MATCH_TYPE к нижнему регистру."""
    assert _build_settings(multi_match_type="BEST_FIELDS").multi_match_type == "best_fields"


def test_multi_match_type_validation_rejects_unknown_value() -> None:
    """Проверяет, что недопустимый OS_MULTI_MATCH_TYPE отклоняется валидацией."""
    with pytest.raises(ValidationError):
        _build_settings(multi_match_type="bad")


def test_min_should_match_accepts_percentage_and_compound_formats() -> None:
    """Проверяет строковые форматы OS_MIN_SHOULD_MATCH, включая процентный и комбинированный."""
    assert _build_settings(min_should_match="70%").min_should_match == "70%"
    assert _build_settings(min_should_match="2<70%").min_should_match == "2<70%"


def test_phrase_field_boosts_parsed_from_json_object() -> None:
    """Проверяет парсинг OS_PHRASE_FIELD_BOOSTS из JSON-строки field->boost."""
    settings = _build_settings(phrase_field_boosts='{"question":6.0,"answer":2.0}')
    assert settings.phrase_field_boosts == {"question": 6.0, "answer": 2.0}


def test_phrase_field_boosts_empty_string_and_empty_object_become_empty_dict() -> None:
    """Проверяет, что пустая строка и пустой объект для OS_PHRASE_FIELD_BOOSTS нормализуются в {}."""
    assert _build_settings(phrase_field_boosts="").phrase_field_boosts == {}
    assert _build_settings(phrase_field_boosts="{}").phrase_field_boosts == {}


def test_phrase_slop_must_be_non_negative() -> None:
    """Проверяет, что OS_PHRASE_SLOP не может быть отрицательным."""
    with pytest.raises(ValidationError):
        _build_settings(phrase_slop=-1)


def test_bool_min_should_match_must_be_positive() -> None:
    """Проверяет, что OS_BOOL_MIN_SHOULD_MATCH должен быть не меньше 1."""
    with pytest.raises(ValidationError):
        _build_settings(bool_min_should_match=0)
