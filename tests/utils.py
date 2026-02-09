import typing as tp

import pandas as pd
import pytest

from app.infrastructure.adapters.open_search import OpenSearchAdapter
from app.infrastructure.storages.milvus import MilvusDatabase


def compare_databases(
    milvus_records: list[dict[str, tp.Any]],
    opensearch_docs: list[dict[str, tp.Any]],
    compare_fields: list[str],
    id_field: str = "ext_id",
) -> None:
    """Сравнивает данные между Milvus и OpenSearch.

    Args:
        milvus_records: данные из Milvus
        opensearch_docs: данные из OpenSearch
        compare_fields: поля для сравнения
        id_field: поле, используемое как идентификатор
    """
    milvus_by_id = {
        record[id_field]: record for record in milvus_records if id_field in record
    }
    opensearch_by_id = {
        doc[id_field]: doc for doc in opensearch_docs if id_field in doc
    }

    milvus_ids = set(milvus_by_id.keys())
    opensearch_ids = set(opensearch_by_id.keys())

    missing_in_opensearch = milvus_ids - opensearch_ids
    missing_in_milvus = opensearch_ids - milvus_ids

    if missing_in_opensearch:
        pytest.fail(
            f"Записи с {id_field} из Milvus отсутствуют в OpenSearch: {missing_in_opensearch}"
        )

    if missing_in_milvus:
        pytest.fail(
            f"Записи с {id_field} из OpenSearch отсутствуют в Milvus: {missing_in_milvus}"
        )

    if len(milvus_ids) != len(opensearch_ids):
        pytest.fail(
            f"Количество уникальных {id_field} не совпадает: "
            f"Milvus={len(milvus_ids)}, OpenSearch={len(opensearch_ids)}"
        )

    common_ids = milvus_ids & opensearch_ids
    for record_id in common_ids:
        milvus_data = milvus_by_id[record_id]
        opensearch_data = opensearch_by_id[record_id]

        for field in compare_fields:
            milvus_value = milvus_data.get(field)
            opensearch_value = opensearch_data.get(field)

            if milvus_value is not None and opensearch_value is not None:
                if str(milvus_value) != str(opensearch_value):
                    pytest.fail(
                        f"Несоответствие данных для {id_field}={record_id}, поле={field}:\n"
                        f"Milvus: {milvus_value}\n"
                        f"OpenSearch: {opensearch_value}"
                    )


def validate_data_counts(
    milvus_count: int,
    opensearch_count: int,
    milvus_records: list[dict[str, tp.Any]],
    opensearch_docs: list[dict[str, tp.Any]],
) -> None:
    """Проверяет совпадение количества записей."""
    if len(milvus_records) != milvus_count:
        pytest.fail(
            f"Milvus вернул {len(milvus_records)} записей вместо {milvus_count}"
        )

    if len(opensearch_docs) != opensearch_count:
        pytest.fail(
            f"OpenSearch вернул {len(opensearch_docs)} записей вместо {opensearch_count}"
        )


def normalize_for_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Нормализует DataFrame для сравнения."""
    df = df.copy()
    df = df.sort_values("ext_id").reset_index(drop=True)

    def convert_numeric(val: tp.Any, col_name: str) -> str:
        if val == "":
            return ""
        try:
            if col_name == "page_id":
                return str(int(float(val)))
            return str(val).strip()
        except (ValueError, TypeError):
            return str(val).strip()

    for col in df.columns:

        def convert_for_column(x: tp.Any, col_name: str = col) -> str:
            return convert_numeric(x, col_name)

        df[col] = df[col].fillna("")
        df[col] = df[col].apply(convert_for_column)

    return df


def compare_with_expected(
    actual_records: list[dict[str, tp.Any]],
    expected_df: pd.DataFrame,
    exclude_fields: list[str] | None = None,
) -> None:
    """Сравнивает фактические данные с ожидаемым DataFrame."""
    if exclude_fields is None:
        exclude_fields = ["row_idx"]

    actual_df = pd.DataFrame(actual_records)
    actual_normalized = normalize_for_comparison(actual_df)

    expected_normalized = normalize_for_comparison(expected_df)

    compare_columns = [
        col
        for col in actual_normalized.columns
        if col in expected_normalized.columns and col not in exclude_fields
    ]

    actual_compare = actual_normalized[compare_columns]
    expected_compare = expected_normalized[compare_columns]

    if len(actual_compare) != len(expected_compare):
        pytest.fail(
            f"Разное количество строк: {len(actual_compare)} vs {len(expected_compare)}"
        )

    mismatches = []
    for idx, (actual_row, expected_row) in enumerate(
        zip(
            actual_compare.itertuples(index=False),
            expected_compare.itertuples(index=False),
            strict=True,
        )
    ):
        for col in compare_columns:
            actual_val = getattr(actual_row, col)
            expected_val = getattr(expected_row, col)

            if actual_val != expected_val:
                mismatches.append(
                    {
                        "index": idx,
                        "column": col,
                        "actual_value": actual_val,
                        "expected_value": expected_val,
                        "ext_id": getattr(actual_row, "ext_id", "unknown"),
                    }
                )

    if mismatches:
        error_msg = "Найдены несовпадения данных:\n"
        for i, mismatch in enumerate(mismatches[:5]):
            error_msg += (
                f"  {i+1}. Строка {mismatch['index']} (ext_id={mismatch['ext_id']}), "
                f"колонка '{mismatch['column']}':\n"
                f"     Фактически: '{mismatch['actual_value']}'\n"
                f"     Ожидалось: '{mismatch['expected_value']}'\n"
            )
        if len(mismatches) > 5:
            error_msg += f"  ... и еще {len(mismatches) - 5} несовпадений\n"

        pytest.fail(error_msg)


async def cleanup_milvus(database: MilvusDatabase, collection_name: str) -> bool:
    """Очищает коллекцию Milvus"""
    if await database._has_collection(collection_name):
        await database._drop_collection(collection_name)
        return True
    return False


async def cleanup_opensearch(adapter: OpenSearchAdapter, index_name: str) -> bool:
    """Очищает индекс OpenSearch"""
    if await adapter.client.indices.exists(index=index_name):
        await adapter.client.indices.delete(index=index_name)
        return True
    return False
