import io

import pandas as pd
from sentence_transformers import SentenceTransformer

from app.common.logger import AISearchLogger
from app.infrastructure.adapters.interfaces import IEduAdapter, IOpenSearchAdapter
from app.infrastructure.storages.interfaces import IVectorDatabase
from app.infrastructure.utils.deduplicated_excel_export import (
    build_deduplicated_knowledge_excel,
    build_export_file_name,
    build_filter_comparison_statistics,
)
from app.infrastructure.utils.prepare_dataframe import (
    combine_validated_sources,
    dedup_by_question_any,
    load_field_mapping,
    prepare_dataframe,
    rename_dataframe,
    split_by_source,
    validate_dataframe,
)
from app.infrastructure.utils.token_filters import MultiValueTokenConfig
from app.infrastructure.utils.universal import cleanup_resources
from app.services.interfaces import IUpdaterService
from app.settings.config import Settings


class UpdaterService(IUpdaterService):
    def __init__(
        self,
        settings: Settings,
        logger: AISearchLogger,
        edu: IEduAdapter,
        milvus: IVectorDatabase,
        os: IOpenSearchAdapter,
    ):
        self.settings = settings
        self.logger = logger
        self.edu = edu
        self.milvus = milvus
        self.os = os
        self.collection_name = settings.milvus.collection_name
        self.model: SentenceTransformer | None = None
        self.field_mapping = load_field_mapping(settings.app.field_mapping_schema_path)
        self.id_column = settings.app.data_unique_id
        # Единый конфиг токенизации для updater/pre_launch/runtime-фильтров.
        self.token_filter_config = MultiValueTokenConfig(
            raw_fields=settings.token_filters.raw_fields,
            token_suffix=settings.token_filters.token_suffix,
            raw_separator=settings.token_filters.raw_separator,
        )

    async def _load_excel_from_edu(self, file_type: str) -> pd.DataFrame:
        if file_type == "vio":
            file_data: io.BytesIO = await self.edu.download_vio_base_file()
        elif file_type == "kb":
            file_data: io.BytesIO = await self.edu.download_kb_base_file()
        else:
            raise ValueError(f"Unknown file_type: {file_type}")
        df = pd.read_excel(file_data)
        self.logger.info(f"Файл '{file_type}' загружен, {len(df)} строк")
        return df

    async def _update_collection_from_df(
        self, df: pd.DataFrame, target_source: str
    ) -> None:
        if df.empty:
            return
        current_source = target_source
        self.logger.info(f"🔄 Обновление данных для источника: {current_source}")

        incoming_ids = df[self.id_column].astype(str).tolist()

        # Используем отдельные проверки для каждой БД
        os_found, os_missing, os_extra = await self.os.ids_exist_by_source_field(
            incoming_ids, source=current_source
        )

        mil_found, mil_missing, mil_extra = await self.milvus.find_existing_ext_ids(
            self.collection_name,
            incoming_ids,
            source_field="source",
            source=current_source,
        )

        incoming_set = set(incoming_ids)

        # Для OpenSearch
        to_delete_os = list(set(os_extra))
        if to_delete_os:
            self.logger.warning(
                f"🗑 OpenSearch: удаляем {len(to_delete_os)} документов..."
            )
            try:
                deleted_count = await self.os.delete_by_ext_ids(to_delete_os)
                self.logger.info(
                    f"✅ OpenSearch: удалено {deleted_count}, ids: {to_delete_os}"
                )
            except Exception as e:
                self.logger.error(f"❌ Ошибка удаления в OpenSearch: {e}")

        # Для Milvus
        to_delete_milvus = list(set(mil_extra))
        if to_delete_milvus:
            self.logger.warning(
                f"🗑 Milvus: удаляем {len(to_delete_milvus)} entities, ids: {to_delete_milvus}"
            )
            try:
                deleted_count = await self.milvus.delete_by_ext_ids(
                    self.collection_name, to_delete_milvus
                )
                self.logger.info(f"✅ Milvus: удалено ~{deleted_count}")
            except Exception as e:
                self.logger.error(f"❌ Ошибка удаления в Milvus: {e}")

        # Определяем, что нужно обновлять/добавлять в OpenSearch
        new_in_os = set(os_missing)
        self.logger.info(
            f"🗑 OpenSearch: новых документов: {len(new_in_os)}, ids: {new_in_os}"
        )
        update_candidates_os = set(os_found)

        # Для найденных в OS проверяем modified_at
        if update_candidates_os:
            self.logger.info("🔎 OpenSearch: сравниваем modified_at...")
            try:
                # Создаем карту modified_at только для найденных записей
                incoming_modified_map_os = {
                    str(r[self.id_column]): (
                        ""
                        if r.get("modified_at") is None
                        else str(r.get("modified_at")).strip()
                    )
                    for r in df[
                        df[self.id_column].astype(str).isin(update_candidates_os)
                    ].to_dict(orient="records")
                }

                os_different = set(
                    await self.os.diff_modified_by_ext_ids(incoming_modified_map_os)
                )
            except Exception as e:
                self.logger.error(f"⚠️ Ошибка при diff_modified_by_ext_ids в OS: {e}")
                os_different = set()
        else:
            os_different = set()
        self.logger.info(
            f"🗑 OpenSearch: измененных документов: {len(os_different)}, ids: {os_different}"
        )
        # Для Milvus аналогично
        new_in_milvus = set(mil_missing)
        self.logger.info(
            f"🗑 Milvus: новых документов: {len(new_in_milvus)}, ids: {new_in_milvus}"
        )
        update_candidates_mil = set(mil_found)

        if update_candidates_mil:
            self.logger.info("🔎 Milvus: сравниваем modified_at...")
            try:
                # Создаем карту modified_at только для найденных записей
                incoming_modified_map_mil = {
                    str(r[self.id_column]): (
                        ""
                        if r.get("modified_at") is None
                        else str(r.get("modified_at")).strip()
                    )
                    for r in df[
                        df[self.id_column].astype(str).isin(update_candidates_mil)
                    ].to_dict(orient="records")
                }

                mil_different = set(
                    await self.milvus.diff_modified_by_ext_ids(
                        self.collection_name, incoming_modified_map_mil
                    )
                )
            except Exception as e:
                self.logger.error(f"⚠️ Ошибка при сравнении modified_at в Milvus: {e}")
                mil_different = set()
        else:
            mil_different = set()
        self.logger.info(
            f"🗑 Milvus: измененных документов: {len(mil_different)}, ids: {mil_different}"
        )
        # Определяем, что нужно создать/обновить в каждой БД отдельно
        to_upsert_os = (new_in_os | os_different) & incoming_set
        to_upsert_mil = (new_in_milvus | mil_different) & incoming_set

        # Объединяем для удобства обработки (уникальные записи)
        to_upsert_all = to_upsert_os | to_upsert_mil

        if not to_upsert_all:
            self.logger.info("✅ Нет новых или изменённых записей для upsert.")
            return

        df_to_upsert = df[df[self.id_column].astype(str).isin(to_upsert_all)].copy()
        if df_to_upsert.empty:
            return

        if self.model is None:
            self.model = SentenceTransformer(
                self.settings.milvus.model_name, local_files_only=True
            )

        docs = df_to_upsert[self.settings.milvus.search_fields].astype(str).tolist()
        metadata = df_to_upsert.to_dict(orient="records")

        self.logger.info(f"⬆️ Подготавливаем upsert для {len(metadata)} записей...")

        try:
            embeddings = await self.milvus.get_embeddings(self.model, docs)
        except Exception as e:
            self.logger.error(f"❌ Ошибка генерации эмбеддингов: {e}")
            raise

        # Upsert в Milvus (только для записей, которые нужно обновить в Milvus)
        mil_metadata = [m for m in metadata if str(m[self.id_column]) in to_upsert_mil]
        if mil_metadata:
            try:
                mil_indices = [
                    i
                    for i, m in enumerate(metadata)
                    if str(m[self.id_column]) in to_upsert_mil
                ]
                mil_embeddings = embeddings[mil_indices]

                await self.milvus.upsert_vectors(
                    self.collection_name, mil_embeddings.tolist(), mil_metadata
                )
                self.logger.info(
                    f"✅ Milvus: upsert выполнен для {len(mil_metadata)} записей"
                )
            except Exception as e:
                self.logger.error(f"❌ Milvus upsert failed: {e}")

        # Upsert в OpenSearch (только для записей, которые нужно обновить в OS)
        os_metadata = [m for m in metadata if str(m[self.id_column]) in to_upsert_os]
        if os_metadata:
            try:
                await self.os.upsert(os_metadata)
                self.logger.info(
                    f"✅ OpenSearch: upsert выполнен для {len(os_metadata)} записей"
                )
            except Exception as e:
                self.logger.error(f"❌ OpenSearch upsert failed: {e}")

        self.logger.info("✅ Обновление коллекции завершено")

    async def update_vio_base(self) -> None:
        vio_df = await self._load_excel_from_edu("vio")
        vio_df_renamed = rename_dataframe(
            vio_df, self.settings.app.field_mapping_schema_path
        )
        vio_df_validated = validate_dataframe(
            vio_df_renamed,
            self.settings.app.field_mapping_schema_path,
            id_column=self.id_column,
        )
        _, _, df_prepared = prepare_dataframe(
            vio_df_validated,
            id_column=self.id_column,
            token_config=self.token_filter_config,
        )
        # prepare_dataframe централизованно добавляет token-поля перед upsert.
        await self._update_collection_from_df(df_prepared, target_source="ВиО")

        cleanup_resources(self.logger)

    async def update_kb_base(self) -> None:
        tp_df = await self._load_excel_from_edu("kb")
        tp_df_renamed = rename_dataframe(
            tp_df, self.settings.app.field_mapping_schema_path
        )
        tp_df_validated = validate_dataframe(
            tp_df_renamed,
            self.settings.app.field_mapping_schema_path,
            id_column=self.id_column,
        )
        _, _, df_prepared = prepare_dataframe(
            tp_df_validated,
            id_column=self.id_column,
            token_config=self.token_filter_config,
        )
        # Те же правила enrichment, что и для ВиО/полной загрузки.
        await self._update_collection_from_df(df_prepared, target_source="ТП")

        cleanup_resources(self.logger)

    async def _export_deduplicated_knowledge_if_enabled(
        self,
        df: pd.DataFrame,
        *,
        statistics_df: pd.DataFrame | None = None,
    ) -> None:
        """Экспортирует дедуплицированные знания в Excel и загружает их в Confluence.

        Метод вызывается только из `update_all` после `prepare_dataframe` и
        обновления обеих source-частей через `_update_collection_from_df`.
        Если флаг `settings.extract_edu.deduplicated_excel_upload_enabled` выключен,
        метод ничего не делает.

        Любая ошибка формирования Excel или загрузки attachment должна логироваться
        вызывающим кодом и не должна прерывать основной процесс обновления Milvus/OpenSearch.
        """
        if not self.settings.extract_edu.deduplicated_excel_upload_enabled:
            self.logger.info(
                "Экспорт дедуплицированного Excel выключен: "
                "EXTRACT_DEDUPLICATED_EXCEL_UPLOAD_ENABLED=false"
            )
            return

        self.logger.info(
            "Экспорт дедуплицированных знаний включен. "
            f"Строк после дедупликации: {len(df)}"
        )
        self.logger.info("Начало формирования Excel-файла с дедуплицированными знаниями")

        file_name = build_export_file_name(
            template=self.settings.extract_edu.deduplicated_excel_file_name_template
        )
        self.logger.info(f"Имя файла для выгрузки: {file_name}")

        excel_bytes = build_deduplicated_knowledge_excel(
            df=df,
            field_mapping=self.field_mapping,
            statistics_df=statistics_df,
            excluded_columns={"row_idx", *self.token_filter_config.token_fields},
        )
        self.logger.info(f"Excel-файл сформирован, размер: {len(excel_bytes)} байт")

        self.logger.info("Начинаем загрузку Excel-вложения в Confluence")
        await self.edu.upload_or_update_attachment_to_edu(
            filename=file_name,
            content=excel_bytes,
            content_type=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
            keep_last_versions=self.settings.extract_edu.deduplicated_excel_keep_versions,
        )

        self.logger.info("Экспорт и загрузка Excel с дедуплицированными знаниями завершены")

    async def update_all(self) -> None:
        df_kb = await self._load_excel_from_edu("kb")
        df_vio = await self._load_excel_from_edu("vio")
        df_kb_renamed = rename_dataframe(
            df_kb, self.settings.app.field_mapping_schema_path
        )
        df_vio_renamed = rename_dataframe(
            df_vio, self.settings.app.field_mapping_schema_path
        )

        df_kb_validated = validate_dataframe(
            df_kb_renamed,
            self.settings.app.field_mapping_schema_path,
            id_column=self.id_column,
        )
        df_vio_validated = validate_dataframe(
            df_vio_renamed,
            self.settings.app.field_mapping_schema_path,
            id_column=self.id_column,
        )

        combined_df = combine_validated_sources(
            [df_kb_validated, df_vio_validated], id_column=self.id_column
        )
        before_filter_df = combined_df.copy()

        combined_deduplicated_df = dedup_by_question_any(combined_df)
        tp_df, vio_df = split_by_source(combined_deduplicated_df)

        prepared_frames: list[pd.DataFrame] = []

        if tp_df is not None and not tp_df.empty:
            _, _, tp_df_prepared = prepare_dataframe(
                tp_df,
                id_column=self.id_column,
                token_config=self.token_filter_config,
            )
        else:
            tp_df_prepared = pd.DataFrame(columns=combined_deduplicated_df.columns)

        await self._update_collection_from_df(tp_df_prepared, target_source="ТП")
        if not tp_df_prepared.empty:
            prepared_frames.append(tp_df_prepared)

        if vio_df is not None and not vio_df.empty:
            _, _, vio_df_prepared = prepare_dataframe(
                vio_df,
                id_column=self.id_column,
                token_config=self.token_filter_config,
            )
        else:
            vio_df_prepared = pd.DataFrame(columns=combined_deduplicated_df.columns)

        await self._update_collection_from_df(vio_df_prepared, target_source="ВиО")
        if not vio_df_prepared.empty:
            prepared_frames.append(vio_df_prepared)

        if prepared_frames:
            prepared_combined_df = pd.concat(prepared_frames, ignore_index=True)
        else:
            prepared_combined_df = pd.DataFrame(columns=combined_deduplicated_df.columns)

        statistics_df = build_filter_comparison_statistics(
            before_df=before_filter_df,
            after_df=prepared_combined_df,
            source_column="source",
            id_column=self.id_column,
            before_count_column="Всего (до фильтрации)",
            after_count_column="Всего (после фильтрации)",
        )

        try:
            await self._export_deduplicated_knowledge_if_enabled(
                prepared_combined_df,
                statistics_df=statistics_df,
            )
        except Exception:
            self.logger.exception(
                "❌ Ошибка формирования или загрузки Excel-файла с дедуплицированными знаниями"
            )

        cleanup_resources(self.logger)
