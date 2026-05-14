import pytest
from pydantic import ValidationError

from app.settings.config import OpenSearchSettings


def _build_settings(**overrides: object) -> OpenSearchSettings:
    """Создаёт валидный OpenSearchSettings для точечных проверок нормализации."""
    base = {
        "host": "localhost",
        "port": 9200,
        "index_name": "kb",
    }
    base.update(overrides)
    return OpenSearchSettings(**base)


def test_os_settings_normalize_operator_min_should_match_and_multi_match_type() -> None:
    """Проверяет нормализацию operator/multi_match_type и сохранение строкового min_should_match."""
    settings = _build_settings(
        operator=" OR ",
        min_should_match=" 70% ",
        multi_match_type=" BEST_FIELDS ",
    )

    assert settings.operator == "or"
    assert settings.min_should_match == "70%"
    assert settings.multi_match_type == "best_fields"


def test_os_settings_invalid_operator_raises_validation_error() -> None:
    """Проверяет, что недопустимый OS operator отклоняется валидацией."""
    with pytest.raises(ValidationError):
        _build_settings(operator="xor")


def test_os_settings_invalid_multi_match_type_raises_validation_error() -> None:
    """Проверяет, что недопустимый multi_match type отклоняется валидацией."""
    with pytest.raises(ValidationError):
        _build_settings(multi_match_type="bad_type")


def test_os_settings_accepts_percentage_min_should_match() -> None:
    """Проверяет поддержку процентного формата minimum_should_match."""
    settings = _build_settings(min_should_match="70%")
    assert settings.min_should_match == "70%"


def test_os_settings_accepts_compound_min_should_match() -> None:
    """Проверяет поддержку составного формата minimum_should_match."""
    settings = _build_settings(min_should_match="2<70%")
    assert settings.min_should_match == "2<70%"
