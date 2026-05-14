"""Тесты валидации search-настроек short/hybrid fusion."""

import pytest
from pydantic import ValidationError

from app.settings.config import HybridSearchSettings, ShortSettings


def test_short_fusion_mode_rrf_normalized() -> None:
    """Проверяет, что short fusion_mode=rrf принимается и остается в нормализованном виде."""
    settings = ShortSettings(
        mode=True,
        mode_limit=3,
        use_opensearch=True,
        fusion_mode="rrf",
    )

    assert settings.fusion_mode == "rrf"


def test_short_fusion_mode_weighted_score_trimmed() -> None:
    """Проверяет strip/lower-нормализацию для short fusion_mode=weighted_score."""
    settings = ShortSettings(
        mode=True,
        mode_limit=3,
        use_opensearch=True,
        fusion_mode=" weighted_score ",
    )

    assert settings.fusion_mode == "weighted_score"


def test_short_fusion_mode_invalid_value_raises_error() -> None:
    """Проверяет, что недопустимый short fusion_mode отклоняется валидатором."""
    with pytest.raises(ValidationError):
        ShortSettings(
            mode=True,
            mode_limit=3,
            use_opensearch=True,
            fusion_mode="bad",
        )


def test_short_rrf_k_must_be_positive() -> None:
    """Проверяет, что short rrf_k=0 отклоняется, а rrf_k=1 является валидным."""
    with pytest.raises(ValidationError):
        ShortSettings(
            mode=True,
            mode_limit=3,
            use_opensearch=True,
            rrf_k=0,
        )

    valid = ShortSettings(
        mode=True,
        mode_limit=3,
        use_opensearch=True,
        rrf_k=1,
    )
    assert valid.rrf_k == 1


def test_hybrid_legacy_weighted_forbidden_with_rrf() -> None:
    """Проверяет, что legacy_weighted отклоняется при fusion_mode=rrf."""
    with pytest.raises(ValidationError, match="HYBRID_FINAL_RANK_MODE=legacy_weighted"):
        HybridSearchSettings(
            intermediate_results_top_k=10,
            fusion_mode="rrf",
            final_rank_mode="legacy_weighted",
        )
