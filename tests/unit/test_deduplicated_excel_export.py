from datetime import UTC, datetime
from io import BytesIO

import pandas as pd

from app.infrastructure.utils.deduplicated_excel_export import (
    build_deduplicated_knowledge_excel,
    build_export_dataframe,
    build_export_file_name,
    build_source_statistics,
)


def test_build_export_dataframe_renames_and_keeps_order() -> None:
    field_mapping = {
        "Источник": "source",
        "ID": "ext_id",
        "Вопрос (clean)": "question",
    }
    source_df = pd.DataFrame(
        [
            {
                "ext_id": "1",
                "question": "Как оформить доступ?",
                "source": "ТП",
                "extra_column": "extra",
            }
        ]
    )

    export_df = build_export_dataframe(source_df, field_mapping)

    assert export_df.columns.tolist() == [
        "Источник",
        "ID",
        "Вопрос (clean)",
        "extra_column",
    ]
    assert export_df.iloc[0]["Источник"] == "ТП"
    assert export_df.iloc[0]["ID"] == "1"


def test_build_source_statistics_counts_unique_ids() -> None:
    source_df = pd.DataFrame(
        {
            "Источник": ["ТП", "ТП", "ТП", "ВиО", "ВиО"],
            "ID": [1, 1, 2, 10, 11],
        }
    )

    stats_df = build_source_statistics(source_df)
    stats_map = dict(zip(stats_df["Источник"], stats_df["Количество уникальных ID"], strict=False))

    assert stats_map == {"ТП": 2, "ВиО": 2}


def test_build_deduplicated_knowledge_excel_returns_expected_sheets() -> None:
    field_mapping = {
        "Источник": "source",
        "ID": "ext_id",
        "Вопрос (clean)": "question",
    }
    source_df = pd.DataFrame(
        [
            {"source": "ТП", "ext_id": "1", "question": "Q1"},
            {"source": "ТП", "ext_id": "1", "question": "Q1 duplicate"},
            {"source": "ВиО", "ext_id": "10", "question": "Q2"},
            {"source": "ВиО", "ext_id": "11", "question": "Q3"},
        ]
    )

    excel_bytes = build_deduplicated_knowledge_excel(source_df, field_mapping)

    assert isinstance(excel_bytes, bytes)
    xls = pd.ExcelFile(BytesIO(excel_bytes))
    assert set(xls.sheet_names) == {"Знания", "Статистика"}

    statistics_df = pd.read_excel(BytesIO(excel_bytes), sheet_name="Статистика")
    stats_map = dict(
        zip(statistics_df["Источник"], statistics_df["Количество уникальных ID"], strict=False)
    )
    assert stats_map == {"ТП": 1, "ВиО": 2}


def test_build_export_file_name_supports_template_and_extension() -> None:
    fixed_ts = datetime(2026, 4, 25, 10, 11, 12, tzinfo=UTC)

    assert (
        build_export_file_name("statistic_{timestamp}.xlsx", timestamp=fixed_ts)
        == "statistic_20260425_101112.xlsx"
    )
    assert (
        build_export_file_name("statistic_{timestamp}", timestamp=fixed_ts)
        == "statistic_20260425_101112.xlsx"
    )
    assert (
        build_export_file_name("statistic.xlsx", timestamp=fixed_ts)
        == "statistic.xlsx"
    )
