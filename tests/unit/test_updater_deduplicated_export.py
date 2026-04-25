from types import SimpleNamespace
from typing import Never
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

import app.services.updater as updater_module
from app.services.updater import UpdaterService


class DummyEduAdapter:
    def __init__(self) -> None:
        self.upload_or_update_attachment_to_edu = AsyncMock()

    async def download_vio_base_file(self) -> Never:
        raise NotImplementedError

    async def download_kb_base_file(self) -> Never:
        raise NotImplementedError

    async def get_attachment_id_from_edu(
        self,
        filename: str,
        page_id: str | None = None,
    ) -> Never:
        raise NotImplementedError

    async def provoke_harvest_to_edu(self, harvest_type: str) -> Never:
        raise NotImplementedError

    async def upload_attachment_to_edu(
        self,
        filename: str,
        content: bytes,
        content_type: str,
        page_id: str | None = None,
    ) -> Never:
        raise NotImplementedError


def _build_settings(upload_enabled: bool) -> SimpleNamespace:
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
        token_filters=SimpleNamespace(
            raw_fields=(),
            token_suffix="_tokens",
            raw_separator=";",
        ),
        extract_edu=SimpleNamespace(
            deduplicated_excel_upload_enabled=upload_enabled,
            deduplicated_excel_file_name_template="statistic_{timestamp}.xlsx",
            deduplicated_excel_keep_versions=5,
        ),
    )


@pytest.mark.asyncio
async def test_export_disabled_does_not_call_upload() -> None:
    edu = DummyEduAdapter()
    logger = MagicMock()
    service = UpdaterService(
        settings=_build_settings(upload_enabled=False),
        logger=logger,
        edu=edu,
        milvus=MagicMock(),
        os=MagicMock(),
    )

    df = pd.DataFrame([{"source": "ТП", "ext_id": "1", "question": "Q"}])

    await service._export_deduplicated_knowledge_if_enabled(df)

    edu.upload_or_update_attachment_to_edu.assert_not_awaited()


@pytest.mark.asyncio
async def test_export_enabled_calls_upload() -> None:
    edu = DummyEduAdapter()
    logger = MagicMock()
    service = UpdaterService(
        settings=_build_settings(upload_enabled=True),
        logger=logger,
        edu=edu,
        milvus=MagicMock(),
        os=MagicMock(),
    )

    df = pd.DataFrame([{"source": "ТП", "ext_id": "1", "question": "Q"}])

    await service._export_deduplicated_knowledge_if_enabled(df)

    edu.upload_or_update_attachment_to_edu.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_all_logs_error_when_export_fails_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    edu = DummyEduAdapter()
    logger = MagicMock()
    service = UpdaterService(
        settings=_build_settings(upload_enabled=False),
        logger=logger,
        edu=edu,
        milvus=MagicMock(),
        os=MagicMock(),
    )

    monkeypatch.setattr(updater_module, "rename_dataframe", lambda df, _: df)
    monkeypatch.setattr(
        updater_module,
        "validate_dataframe",
        lambda df, _1, id_column: df,
    )
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
    monkeypatch.setattr(
        updater_module,
        "prepare_dataframe",
        lambda df, id_column, token_config: ([], [], df),
    )
    monkeypatch.setattr(updater_module, "cleanup_resources", lambda logger: None)

    service._load_excel_from_edu = AsyncMock(
        side_effect=[
            pd.DataFrame([{"source": "ТП", "ext_id": "1", "question": "Q"}]),
            pd.DataFrame([{"source": "ВиО", "ext_id": "10", "question": "Q2"}]),
        ]
    )
    service._export_deduplicated_knowledge_if_enabled = AsyncMock(
        side_effect=RuntimeError("upload failed")
    )
    service._update_collection_from_df = AsyncMock()

    await service.update_all()

    logger.exception.assert_called_once()
    assert service._update_collection_from_df.await_count == 2
