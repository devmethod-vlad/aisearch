import pandas as pd

from app.infrastructure.utils.prepare_dataframe import prepare_dataframe
from app.infrastructure.utils.exact_filters import ExactFilterConfig
from app.infrastructure.utils.token_filters import MultiValueTokenConfig


def test_prepare_dataframe_adds_token_and_exact_fields() -> None:
    """Проверяет добавление token- и exact-полей без изменения raw-значений."""
    df = pd.DataFrame(
        [
            {
                "source": "ТП",
                "ext_id": "1",
                "question": "q",
                "answer": "a",
                "actual": "да",
                "space": "x",
                "role": "Врач;Медсестра",
                "product": "ЭМИАС;ЛИС",
                "component": "Назначения;Расписания;НАЗНАЧЕНИЯ",
                "modified_at": "2026-01-01",
            }
        ]
    )

    _, metadata, prepared = prepare_dataframe(
        df,
        id_column="ext_id",
        token_config=MultiValueTokenConfig(
            raw_fields=("role", "product", "component"),
            token_suffix="_tokens",
            raw_separator=";",
        ),
        exact_filter_config=ExactFilterConfig(
            raw_fields=("source", "actual", "second_line"),
            field_suffix="_filter",
        ),
    )

    assert prepared.iloc[0]["role_tokens"] == ["врач", "медсестра"]
    assert prepared.iloc[0]["product_tokens"] == ["эмиас", "лис"]
    assert prepared.iloc[0]["component"] == "Назначения;Расписания;НАЗНАЧЕНИЯ"
    assert prepared.iloc[0]["component_tokens"] == ["назначения", "расписания"]
    assert metadata[0]["role_tokens"] == ["врач", "медсестра"]
    assert metadata[0]["component_tokens"] == ["назначения", "расписания"]

    assert prepared.iloc[0]["source_filter"] == "тп"
    assert prepared.iloc[0]["actual_filter"] == "да"
    assert prepared.iloc[0]["source"] == "ТП"
