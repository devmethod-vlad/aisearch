from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

import app.services.updater as updater_module
from app.services.updater import UpdaterService


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(
            field_mapping_schema_path="app/settings/field_mapping.json",
            data_unique_id="ext_id",
        ),
        milvus=SimpleNamespace(
            collection_name="kb_default",
            model_name="test-model",
            search_fields="question",
        ),
        opensearch=SimpleNamespace(index_name="kb_index"),
        token_filters=SimpleNamespace(
            raw_fields=(),
            token_suffix="_tokens",
            raw_separator=";",
        ),
        extract_edu=SimpleNamespace(
            deduplicated_excel_upload_enabled=False,
            deduplicated_excel_file_name_template="dedup_{timestamp}.xlsx",
            deduplicated_excel_keep_versions=3,
        ),
    )


def _service() -> UpdaterService:
    service = UpdaterService(
        settings=_settings(),
        logger=MagicMock(),
        edu=MagicMock(),
        milvus=MagicMock(),
        os=MagicMock(),
        redis=AsyncMock(),
    )
    service.model = MagicMock()
    return service


@pytest.mark.asyncio
async def test_update_collection_from_df_returns_false_when_no_real_changes() -> None:
    service = _service()
    df = pd.DataFrame([{"ext_id": "1", "modified_at": "2024-01-01", "question": "Q"}])
    service.os.ids_exist_by_source_field = AsyncMock(return_value=(["1"], [], []))
    service.milvus.find_existing_ext_ids = AsyncMock(return_value=(["1"], [], []))
    service.os.diff_modified_by_ext_ids = AsyncMock(return_value=[])
    service.milvus.diff_modified_by_ext_ids = AsyncMock(return_value=[])

    changed = await service._update_collection_from_df(df, target_source="ТП")

    assert changed is False


@pytest.mark.asyncio
async def test_update_collection_from_df_returns_true_on_successful_delete() -> None:
    service = _service()
    df = pd.DataFrame([{"ext_id": "1", "modified_at": "2024-01-01", "question": "Q"}])
    service.os.ids_exist_by_source_field = AsyncMock(return_value=(["1"], [], ["999"]))
    service.milvus.find_existing_ext_ids = AsyncMock(return_value=(["1"], [], []))
    service.os.delete_by_ext_ids = AsyncMock(return_value=1)
    service.os.diff_modified_by_ext_ids = AsyncMock(return_value=[])
    service.milvus.diff_modified_by_ext_ids = AsyncMock(return_value=[])

    changed = await service._update_collection_from_df(df, target_source="ТП")

    assert changed is True


@pytest.mark.asyncio
async def test_update_collection_from_df_returns_true_on_successful_upsert() -> None:
    service = _service()
    df = pd.DataFrame([{"ext_id": "1", "modified_at": "2024-01-01", "question": "Q"}])
    service.os.ids_exist_by_source_field = AsyncMock(return_value=([], ["1"], []))
    service.milvus.find_existing_ext_ids = AsyncMock(return_value=(["1"], [], []))
    service.os.diff_modified_by_ext_ids = AsyncMock(return_value=[])
    service.milvus.diff_modified_by_ext_ids = AsyncMock(return_value=[])
    service.milvus.get_embeddings = AsyncMock(return_value=np.array([[0.1, 0.2]]))
    service.os.upsert = AsyncMock(return_value=None)

    changed = await service._update_collection_from_df(df, target_source="ТП")

    assert changed is True


@pytest.mark.asyncio
async def test_update_all_bumps_once_when_any_part_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    bump_mock = AsyncMock()
    monkeypatch.setattr(updater_module, "bump_search_data_version", bump_mock)
    monkeypatch.setattr(updater_module, "rename_dataframe", lambda df, _: df)
    monkeypatch.setattr(updater_module, "validate_dataframe", lambda df, _1, id_column: df)
    monkeypatch.setattr(
        updater_module,
        "combine_validated_sources",
        lambda dataframes, id_column: pd.concat(dataframes, ignore_index=True),
    )
    monkeypatch.setattr(updater_module, "dedup_by_question_any", lambda df: df)
    monkeypatch.setattr(
        updater_module,
        "split_by_source",
        lambda df: (df[df["source"] == "ТП"], df[df["source"] == "ВиО"]),
    )
    monkeypatch.setattr(updater_module, "prepare_dataframe", lambda df, id_column, token_config: ([], [], df))
    monkeypatch.setattr(updater_module, "cleanup_resources", lambda logger: None)

    service._load_excel_from_edu = AsyncMock(
        side_effect=[
            pd.DataFrame([{"source": "ТП", "ext_id": "1", "question": "Q"}]),
            pd.DataFrame([{"source": "ВиО", "ext_id": "2", "question": "Q2"}]),
        ]
    )
    service._update_collection_from_df = AsyncMock(side_effect=[True, False])
    service._export_deduplicated_knowledge_if_enabled = AsyncMock()

    await service.update_all()

    bump_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_all_does_not_bump_without_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    bump_mock = AsyncMock()
    monkeypatch.setattr(updater_module, "bump_search_data_version", bump_mock)
    monkeypatch.setattr(updater_module, "rename_dataframe", lambda df, _: df)
    monkeypatch.setattr(updater_module, "validate_dataframe", lambda df, _1, id_column: df)
    monkeypatch.setattr(
        updater_module,
        "combine_validated_sources",
        lambda dataframes, id_column: pd.concat(dataframes, ignore_index=True),
    )
    monkeypatch.setattr(updater_module, "dedup_by_question_any", lambda df: df)
    monkeypatch.setattr(
        updater_module,
        "split_by_source",
        lambda df: (df[df["source"] == "ТП"], df[df["source"] == "ВиО"]),
    )
    monkeypatch.setattr(updater_module, "prepare_dataframe", lambda df, id_column, token_config: ([], [], df))
    monkeypatch.setattr(updater_module, "cleanup_resources", lambda logger: None)

    service._load_excel_from_edu = AsyncMock(
        side_effect=[
            pd.DataFrame([{"source": "ТП", "ext_id": "1", "question": "Q"}]),
            pd.DataFrame([{"source": "ВиО", "ext_id": "2", "question": "Q2"}]),
        ]
    )
    service._update_collection_from_df = AsyncMock(side_effect=[False, False])
    service._export_deduplicated_knowledge_if_enabled = AsyncMock()

    await service.update_all()

    bump_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_kb_and_vio_bump_only_on_real_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    bump_mock = AsyncMock()
    monkeypatch.setattr(updater_module, "bump_search_data_version", bump_mock)
    monkeypatch.setattr(updater_module, "rename_dataframe", lambda df, _: df)
    monkeypatch.setattr(updater_module, "validate_dataframe", lambda df, _1, id_column: df)
    monkeypatch.setattr(updater_module, "prepare_dataframe", lambda df, id_column, token_config: ([], [], df))
    monkeypatch.setattr(updater_module, "cleanup_resources", lambda logger: None)

    service._load_excel_from_edu = AsyncMock(
        side_effect=[
            pd.DataFrame([{"source": "ТП", "ext_id": "1", "question": "Q"}]),
            pd.DataFrame([{"source": "ВиО", "ext_id": "2", "question": "Q2"}]),
        ]
    )
    service._update_collection_from_df = AsyncMock(side_effect=[True, False])

    await service.update_kb_base()
    await service.update_vio_base()

    bump_mock.assert_awaited_once()
