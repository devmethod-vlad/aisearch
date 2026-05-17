"""Тесты валидации HYBRID-настроек для финального ранжирования."""

from typing import Any

import pytest
from pydantic import ValidationError

from app.settings.config import HybridSearchSettings


def _base_kwargs(**overrides: Any) -> dict[str, Any]:
    """Возвращает базовый набор валидных аргументов для HybridSearchSettings."""
    kwargs: dict[str, Any] = {
        "intermediate_results_top_k": 10,
        "fusion_mode": "weighted_score",
    }
    kwargs.update(overrides)
    return kwargs


def test_final_rank_mode_accepts_all_supported_values() -> None:
    """Проверяет допустимые значения final_rank_mode."""
    for mode in ("fusion_only", "ce_final", "ce_blend", "legacy_weighted"):
        settings = HybridSearchSettings(**_base_kwargs(), final_rank_mode=mode)
        assert settings.final_rank_mode == mode


def test_legacy_weighted_valid_with_weighted_score() -> None:
    """Проверяет, что legacy_weighted валиден в паре с weighted_score."""
    settings = HybridSearchSettings(
        **_base_kwargs(
            fusion_mode="weighted_score",
            final_rank_mode="legacy_weighted",
        )
    )
    assert settings.final_rank_mode == "legacy_weighted"


def test_legacy_weighted_invalid_with_rrf() -> None:
    """Проверяет, что legacy_weighted запрещен при fusion_mode=rrf."""
    with pytest.raises(ValidationError, match="HYBRID_FINAL_RANK_MODE=legacy_weighted"):
        HybridSearchSettings(
            **_base_kwargs(
                fusion_mode="rrf",
                final_rank_mode="legacy_weighted",
            )
        )


def test_invalid_final_rank_mode_raises() -> None:
    """Проверяет ошибку на невалидном final_rank_mode."""
    with pytest.raises(ValidationError, match="final_rank_mode должен быть одним из"):
        HybridSearchSettings(**_base_kwargs(), final_rank_mode="bad")


def test_invalid_final_fusion_norm_raises() -> None:
    """Проверяет ошибку на невалидном final_fusion_norm."""
    with pytest.raises(ValidationError, match="final_fusion_norm должен быть"):
        HybridSearchSettings(**_base_kwargs(), final_fusion_norm="bad")


def test_invalid_final_ce_score_raises() -> None:
    """Проверяет ошибку на невалидном final_ce_score."""
    with pytest.raises(ValidationError, match="final_ce_score должен быть"):
        HybridSearchSettings(**_base_kwargs(), final_ce_score="bad")


def test_negative_final_w_fusion_raises() -> None:
    """Проверяет ошибку на отрицательном final_w_fusion."""
    with pytest.raises(ValidationError, match="final_w_fusion не может быть"):
        HybridSearchSettings(**_base_kwargs(), final_w_fusion=-0.1)


def test_negative_final_w_ce_raises() -> None:
    """Проверяет ошибку на отрицательном final_w_ce."""
    with pytest.raises(ValidationError, match="final_w_ce не может быть"):
        HybridSearchSettings(**_base_kwargs(), final_w_ce=-0.1)
