import pandas as pd

from app.infrastructure.utils.prepare_dataframe import prepare_dataframe
from app.infrastructure.utils.token_filters import MultiValueTokenConfig


def test_prepare_dataframe_adds_token_fields() -> None:
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
                "modified_at": "2026-01-01",
            }
        ]
    )

    _, metadata, prepared = prepare_dataframe(
        df,
        id_column="ext_id",
        token_config=MultiValueTokenConfig(
            raw_fields=("role", "product"),
            token_suffix="_tokens",
            raw_separator=";",
        ),
    )

    assert prepared.iloc[0]["role_tokens"] == ["врач", "медсестра"]
    assert prepared.iloc[0]["product_tokens"] == ["эмиас", "лис"]
    assert metadata[0]["role_tokens"] == ["врач", "медсестра"]
