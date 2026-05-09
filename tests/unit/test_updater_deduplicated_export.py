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
        exact_filters=SimpleNamespace(
            raw_fields=(),
            field_suffix="_filter",
        ),
        opensearch=SimpleNamespace(index_name="kb_index"),
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
        redis=AsyncMock(),
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
        redis=AsyncMock(),
    )

    df = pd.DataFrame([{"source": "ТП", "ext_id": "1", "question": "Q"}])

    await service._export_deduplicated_knowledge_if_enabled(df)

    edu.upload_or_update_attachment_to_edu.assert_awaited_once()


@pytest.mark.asyncio
async def test_export_enabled_accepts_optional_statistics_df() -> None:
    edu = DummyEduAdapter()
    logger = MagicMock()
    service = UpdaterService(
        settings=_build_settings(upload_enabled=True),
        logger=logger,
        edu=edu,
        milvus=MagicMock(),
        os=MagicMock(),
        redis=AsyncMock(),
    )

    df = pd.DataFrame([{"source": "ТП", "ext_id": "1", "question": "Q"}])
    statistics_df = pd.DataFrame(
        [{"Источник": "ТП", "Всего (до фильтрации)": 1, "Всего (после фильтрации)": 1}]
    )

    await service._export_deduplicated_knowledge_if_enabled(df, statistics_df=statistics_df)

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
        redis=AsyncMock(),
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
        lambda df, id_column, token_config, exact_filter_config=None: ([], [], df),
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


@pytest.mark.asyncio
async def test_update_all_calls_export_after_both_updates(
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
        redis=AsyncMock(),
    )

    events: list[str] = []
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
    monkeypatch.setattr(updater_module, "prepare_dataframe", lambda df, id_column, token_config, exact_filter_config=None: ([], [], df))
    monkeypatch.setattr(updater_module, "cleanup_resources", lambda logger: None)

    service._load_excel_from_edu = AsyncMock(
        side_effect=[
            pd.DataFrame([{"source": "ТП", "ext_id": "1", "question": "Q"}]),
            pd.DataFrame([{"source": "ВиО", "ext_id": "10", "question": "Q2"}]),
        ]
    )

    async def update_side_effect(df: pd.DataFrame, target_source: str) -> None:
        events.append(f"update:{target_source}")

    async def export_side_effect(df: pd.DataFrame, *, statistics_df: pd.DataFrame | None = None) -> None:
        events.append("export")

    service._update_collection_from_df = AsyncMock(side_effect=update_side_effect)
    service._export_deduplicated_knowledge_if_enabled = AsyncMock(side_effect=export_side_effect)

    await service.update_all()

    assert events == ["update:ТП", "update:ВиО", "export"]


@pytest.mark.asyncio
async def test_update_all_passes_combined_prepared_df_to_export(
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
        redis=AsyncMock(),
    )

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
    monkeypatch.setattr(updater_module, "cleanup_resources", lambda logger: None)

    tp_prepared = pd.DataFrame([{"source": "ТП", "ext_id": "tp1", "question": "tp"}])
    vio_prepared = pd.DataFrame([{"source": "ВиО", "ext_id": "vio1", "question": "vio"}])

    def prepare_side_effect(
        df: pd.DataFrame,
        id_column: str,
        token_config: object,
        exact_filter_config: object | None = None,
    ) -> tuple[list[object], list[object], pd.DataFrame]:
        if df["source"].iloc[0] == "ТП":
            return ([], [], tp_prepared)
        return ([], [], vio_prepared)

    monkeypatch.setattr(updater_module, "prepare_dataframe", prepare_side_effect)

    service._load_excel_from_edu = AsyncMock(
        side_effect=[
            pd.DataFrame([{"source": "ТП", "ext_id": "1", "question": "Q"}]),
            pd.DataFrame([{"source": "ВиО", "ext_id": "10", "question": "Q2"}]),
        ]
    )
    service._update_collection_from_df = AsyncMock()
    service._export_deduplicated_knowledge_if_enabled = AsyncMock()

    await service.update_all()

    exported_df = service._export_deduplicated_knowledge_if_enabled.await_args.args[0]
    assert set(exported_df["ext_id"].tolist()) == {"tp1", "vio1"}


@pytest.mark.asyncio
async def test_update_all_passes_filter_comparison_statistics_to_export(
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
        redis=AsyncMock(),
    )

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
    monkeypatch.setattr(updater_module, "cleanup_resources", lambda logger: None)

    def prepare_side_effect(
        df: pd.DataFrame,
        id_column: str,
        token_config: object,
        exact_filter_config: object | None = None,
    ) -> tuple[list[object], list[object], pd.DataFrame]:
        if df["source"].iloc[0] == "ТП":
            return ([], [], pd.DataFrame([{"source": "ТП", "ext_id": "1", "question": "Q"}]))
        return ([], [], pd.DataFrame([{"source": "ВиО", "ext_id": "10", "question": "Q2-filtered"}]))

    monkeypatch.setattr(updater_module, "prepare_dataframe", prepare_side_effect)

    service._load_excel_from_edu = AsyncMock(
        side_effect=[
            pd.DataFrame(
                [
                    {"source": "ТП", "ext_id": "1", "question": "Q"},
                    {"source": "ТП", "ext_id": "2", "question": "Q new"},
                ]
            ),
            pd.DataFrame(
                [
                    {"source": "ВиО", "ext_id": "10", "question": "Q2"},
                    {"source": "ВиО", "ext_id": "11", "question": "Q3"},
                ]
            ),
        ]
    )
    service._update_collection_from_df = AsyncMock()
    service._export_deduplicated_knowledge_if_enabled = AsyncMock()

    await service.update_all()

    statistics_df = service._export_deduplicated_knowledge_if_enabled.await_args.kwargs["statistics_df"]
    assert statistics_df.columns.tolist() == [
        "Источник",
        "Всего (до фильтрации)",
        "Всего (после фильтрации)",
    ]
    stats_map = {
        row["Источник"]: (
            row["Всего (до фильтрации)"],
            row["Всего (после фильтрации)"],
        )
        for _, row in statistics_df.iterrows()
    }
    assert stats_map == {"ТП": (2, 1), "ВиО": (2, 1)}
