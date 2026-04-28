from datetime import UTC, datetime
from io import BytesIO

import pandas as pd

from app.infrastructure.utils.deduplicated_excel_export import (
    build_deduplicated_knowledge_excel,
    build_export_dataframe,
    build_export_file_name,
    build_filter_comparison_statistics,
    build_source_statistics,
)


def test_build_deduplicated_knowledge_excel_handles_missing_values() -> None:
    field_mapping = {
        "Источник": "source",
        "ID": "ext_id",
        "Вопрос (clean)": "question",
        "Ответ (clean)": "answer",
    }

    df = pd.DataFrame(
        [
            {
                "source": "ТП",
                "ext_id": "1",
                "question": "Вопрос 1",
                "answer": pd.NA,
            },
            {
                "source": "ВиО",
                "ext_id": "2",
                "question": None,
                "answer": "Ответ 2",
            },
        ]
    )

    excel_bytes = build_deduplicated_knowledge_excel(
        df=df,
        field_mapping=field_mapping,
    )

    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 0

    excel_file = pd.ExcelFile(BytesIO(excel_bytes))
    assert set(excel_file.sheet_names) == {"Знания", "Статистика"}


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


def test_build_source_statistics_with_internal_columns() -> None:
    source_df = pd.DataFrame(
        {
            "source": ["ТП", "ТП", "ВиО"],
            "ext_id": [1, 2, 2],
        }
    )

    stats_df = build_source_statistics(
        source_df,
        source_column="source",
        id_column="ext_id",
        count_column="Всего (до фильтрации)",
    )

    assert stats_df.columns.tolist() == ["Источник", "Всего (до фильтрации)"]
    stats_map = dict(zip(stats_df["Источник"], stats_df["Всего (до фильтрации)"], strict=False))
    assert stats_map == {"ТП": 2, "ВиО": 1}


def test_build_filter_comparison_statistics_counts_unique_and_fills_zero() -> None:
    before_df = pd.DataFrame(
        [
            {"source": "ТП", "ext_id": 1},
            {"source": "ТП", "ext_id": 1},
            {"source": "ТП", "ext_id": 2},
            {"source": "ВиО", "ext_id": 10},
            {"source": "ВиО", "ext_id": 11},
        ]
    )
    after_df = pd.DataFrame(
        [
            {"source": "ТП", "ext_id": 2},
            {"source": "ВиО", "ext_id": 11},
            {"source": "Новый", "ext_id": 100},
        ]
    )

    stats_df = build_filter_comparison_statistics(before_df=before_df, after_df=after_df)
    assert stats_df.columns.tolist() == [
        "Источник",
        "Всего (до фильтрации)",
        "Всего (после фильтрации)",
    ]
    stats_map = {
        row["Источник"]: (
            row["Всего (до фильтрации)"],
            row["Всего (после фильтрации)"],
        )
        for _, row in stats_df.iterrows()
    }
    assert stats_map == {"ТП": (2, 1), "ВиО": (2, 1), "Новый": (0, 1)}


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


def test_build_deduplicated_knowledge_excel_uses_passed_statistics_df() -> None:
    field_mapping = {
        "Источник": "source",
        "ID": "ext_id",
        "Вопрос (clean)": "question",
    }
    source_df = pd.DataFrame(
        [{"source": "ТП", "ext_id": "1", "question": "Q1"}]
    )
    statistics_df = pd.DataFrame(
        [
            {
                "Источник": "ТП",
                "Всего (до фильтрации)": 10,
                "Всего (после фильтрации)": 9,
            }
        ]
    )

    excel_bytes = build_deduplicated_knowledge_excel(
        source_df,
        field_mapping,
        statistics_df=statistics_df,
    )
    result_df = pd.read_excel(BytesIO(excel_bytes), sheet_name="Статистика")

    assert result_df.columns.tolist() == [
        "Источник",
        "Всего (до фильтрации)",
        "Всего (после фильтрации)",
    ]


def test_build_export_dataframe_excludes_service_columns() -> None:
    field_mapping = {
        "Источник": "source",
        "ID": "ext_id",
    }
    source_df = pd.DataFrame(
        [
            {
                "source": "ТП",
                "ext_id": "1",
                "row_idx": 0,
                "role_tokens": "admin",
                "extra_column": "keep",
            }
        ]
    )

    export_df = build_export_dataframe(
        source_df,
        field_mapping,
        excluded_columns={"row_idx", "role_tokens"},
    )

    assert "row_idx" not in export_df.columns
    assert "role_tokens" not in export_df.columns
    assert "extra_column" in export_df.columns


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
