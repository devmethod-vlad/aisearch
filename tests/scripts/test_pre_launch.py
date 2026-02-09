import subprocess
import sys
from pathlib import Path

import pytest

from app.infrastructure.adapters.open_search import OpenSearchAdapter
from app.infrastructure.storages.milvus import MilvusDatabase
from app.settings.config import MilvusSettings, OpenSearchSettings
from tests.utils import compare_databases, validate_data_counts


@pytest.mark.asyncio
async def test_pre_launch(
    pre_launch_env_vars: dict[str, str],
    milvus_database: MilvusDatabase,
    opensearch_adapter: OpenSearchAdapter,
    test_milvus_settings: MilvusSettings,
    test_opensearch_settings: OpenSearchSettings,
    clean_databases: None,
) -> None:
    """End-to-end тест pre_launch.py"""
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

    assert result.returncode == 0, f"pre_launch failed: {result.stderr}"

    os_index = test_opensearch_settings.index_name
    milvus_collection = test_milvus_settings.collection_name

    assert await opensearch_adapter.client.indices.exists(index=os_index)

    collections = await milvus_database._list_collections()
    assert milvus_collection in collections

    os_doc_count = (await opensearch_adapter.client.count(index=os_index)).get(
        "count", 0
    )
    collection_stats = await milvus_database._get_collection_stats(milvus_collection)
    milvus_row_count = int(collection_stats.get("row_count", 0))

    assert os_doc_count > 0
    assert milvus_row_count > 0
    assert milvus_row_count == os_doc_count

    compare_fields = sorted(
        set(test_milvus_settings.output_fields)
        | set(test_opensearch_settings.output_fields)
    )

    os_docs = await opensearch_adapter.fetch_existing(size=os_doc_count)
    milvus_records = await milvus_database.fetch_existing(
        collection_name=milvus_collection, output_fields=compare_fields
    )

    validate_data_counts(milvus_row_count, os_doc_count, milvus_records, os_docs)
    compare_databases(
        milvus_records=milvus_records,
        opensearch_docs=os_docs,
        compare_fields=compare_fields,
        id_field="ext_id",
    )


@pytest.mark.asyncio
async def test_pre_launch_no_recreate(
    pre_launch_env_vars: dict[str, str],
    milvus_database: MilvusDatabase,
    opensearch_adapter: OpenSearchAdapter,
    test_milvus_settings: MilvusSettings,
    test_opensearch_settings: OpenSearchSettings,
    clean_databases: None,
) -> None:
    """Тест pre_launch.py без пересоздания коллекций и индексов"""
    from sentence_transformers import SentenceTransformer

    milvus_collection = test_milvus_settings.collection_name
    os_index = test_opensearch_settings.index_name

    test_documents = ["Тест1", "Тест2", "Тест3"]
    test_metadata = [
        {"ext_id": f"test_{i}", "question": f"Q{i}", "answer": f"A{i}"}
        for i in range(1, 4)
    ]

    model = SentenceTransformer(test_milvus_settings.model_name, local_files_only=True)

    await milvus_database.initialize_collection(
        collection_name=milvus_collection,
        model=model,
        documents=test_documents,
        metadata=test_metadata,
    )

    await opensearch_adapter.create_index()
    await opensearch_adapter.build_index_with_data(data=test_metadata)

    env_vars = pre_launch_env_vars.copy()
    env_vars.update({"APP_RECREATE_DATA": "false"})

    script_path = Path(__file__).parent.parent.parent / "pre_launch.py"
    result = subprocess.run(
        [sys.executable, str(script_path)],
        env=env_vars,
        capture_output=True,
        timeout=600,
        encoding="utf-8",
        errors="replace",
        text=True,
        check=False,
    )

    assert result.returncode == 0

    final_milvus_stats = await milvus_database._get_collection_stats(milvus_collection)
    final_milvus_count = int(final_milvus_stats.get("row_count", 0))
    final_os_count = (await opensearch_adapter.client.count(index=os_index)).get(
        "count", 0
    )

    assert final_milvus_count == 3
    assert final_os_count == 3
