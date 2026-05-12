"""Тесты валидации search-настроек short/hybrid fusion."""

import pytest
from pydantic import ValidationError

from app.settings.config import ShortSettings


def test_short_fusion_mode_rrf_normalized() -> None:
    """Проверяет, что short fusion_mode=rrf принимается и остается в нормализованном виде."""
    settings = ShortSettings(
        mode=True,
        mode_limit=3,
        use_opensearch=True,
        use_reranker=False,
        use_hybrid=True,
        fusion_mode="rrf",
    )

    assert settings.fusion_mode == "rrf"


def test_short_fusion_mode_weighted_score_trimmed() -> None:
    """Проверяет strip/lower-нормализацию для short fusion_mode=weighted_score."""
    settings = ShortSettings(
        mode=True,
        mode_limit=3,
        use_opensearch=True,
        use_reranker=False,
        use_hybrid=True,
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
            use_reranker=False,
            use_hybrid=True,
            fusion_mode="bad",
        )


def test_short_rrf_k_must_be_positive() -> None:
    """Проверяет, что short rrf_k=0 отклоняется, а rrf_k=1 является валидным."""
    with pytest.raises(ValidationError):
        ShortSettings(
            mode=True,
            mode_limit=3,
            use_opensearch=True,
            use_reranker=False,
            use_hybrid=True,
            rrf_k=0,
        )

    valid = ShortSettings(
        mode=True,
        mode_limit=3,
        use_opensearch=True,
        use_reranker=False,
        use_hybrid=True,
        rrf_k=1,
    )
    assert valid.rrf_k == 1
