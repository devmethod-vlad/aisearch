import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from app.infrastructure.adapters.open_search import OpenSearchAdapter
from app.infrastructure.storages.milvus import MilvusDatabase
from app.services.updater import UpdaterService
from app.settings.config import MilvusSettings, OpenSearchSettings
from tests.utils import (
    compare_databases,
    compare_with_expected,
    validate_data_counts,
)


@pytest.mark.asyncio
async def test_prelaunch_and_updater(
    pre_launch_env_vars: dict[str, str],
    milvus_database: MilvusDatabase,
    opensearch_adapter: OpenSearchAdapter,
    test_milvus_settings: MilvusSettings,
    test_opensearch_settings: OpenSearchSettings,
    updater_service: UpdaterService,
    prelaunch_updater_correct_result_df: pd.DataFrame,
    clean_databases: None,
) -> None:
    milvus_collection = test_milvus_settings.collection_name
    os_index = test_opensearch_settings.index_name

    script_path = Path(__file__).parent.parent.parent / "pre_launch.py"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        env=pre_launch_env_vars,
        capture_output=True,
        timeout=600,
        encoding="utf-8",
        errors="replace",
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"pre_launch завершился с ошибкой: {result.stderr}"

    assert await opensearch_adapter.client.indices.exists(index=os_index)

    collections = await milvus_database._list_collections()
    assert milvus_collection in collections

    await updater_service.update_all()
    await milvus_database._flush(milvus_collection)

    final_os_count = (await opensearch_adapter.client.count(index=os_index)).get(
        "count", 0
    )

    compare_fields = sorted(
        set(test_milvus_settings.output_fields)
        | set(test_opensearch_settings.output_fields)
    )

    milvus_records = await milvus_database.fetch_existing(
        collection_name=milvus_collection,
        output_fields=compare_fields,
    )

    actual_milvus_count = len(milvus_records)

    assert actual_milvus_count > 0
    assert final_os_count > 0
    assert actual_milvus_count == final_os_count

    expected_count = len(prelaunch_updater_correct_result_df)
    assert actual_milvus_count == expected_count

    os_docs = await opensearch_adapter.fetch_existing(size=final_os_count)

    validate_data_counts(actual_milvus_count, final_os_count, milvus_records, os_docs)

    compare_databases(
        milvus_records=milvus_records,
        opensearch_docs=os_docs,
        compare_fields=compare_fields,
        id_field="ext_id",
    )

    compare_with_expected(
        actual_records=milvus_records,
        expected_df=prelaunch_updater_correct_result_df,
        exclude_fields=["row_idx"],
    )
