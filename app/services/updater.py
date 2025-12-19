import io

import pandas as pd
from sentence_transformers import SentenceTransformer

from app.common.logger import AISearchLogger
from app.infrastructure.adapters.interfaces import IEduAdapter, IOpenSearchAdapter
from app.infrastructure.storages.interfaces import IVectorDatabase
from app.infrastructure.utils.prepare_dataframe import (
    dedup_by_question_any,
    prepare_dataframe,
    rename_dataframe
)
from app.infrastructure.utils.universal import cleanup_resources
from app.services.interfaces import IUpdaterService
from app.settings.config import Settings


class UpdaterService(IUpdaterService):
    FIELD_MAPPING = {
        "Источник": "source",
        "ID": "ext_id",
        "ID страницы": "page_id",
        "Актуально": "actual",
        "2 линия": "second_line",
        "Роль": "role",
        "Продукт": "product",
        "Пространство": "space",
        "Компонент": "component",
        "Вопрос (markdown)": "question_md",
        "Вопрос (clean)": "question",
        "Анализ ошибки (markdown)": "analysis_md",
        "Анализ ошибки (clean)": "analysis",
        "Ответ (markdown)": "answer_md",
        "Ответ (clean)": "answer",
        "Для пользователя": "for_user",
        "Jira": "jira",
        "Обновлено": "modified_at",
    }

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

    def _prepare_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        _, _, df_prepared = prepare_dataframe(
            df=df,
            logger=self.logger
        )

        return df_prepared

    async def _update_collection_from_df(
        self, df: pd.DataFrame, target_source: str
    ) -> None:
        if df.empty:
            return

        current_source = target_source
        self.logger.info(f"🔄 Обновление данных для источника: {current_source}")

        incoming_ext_ids = df["ext_id"].astype(str).tolist()

        # Используем отдельные проверки для каждой БД
        os_found, os_missing, os_extra = self.os.ids_exist_by_source_field(
            incoming_ext_ids, source=current_source
        )

        mil_found, mil_missing, mil_extra = await self.milvus.find_existing_ext_ids(
            self.collection_name,
            incoming_ext_ids,
            source_field="source",
            source=current_source,
        )

        incoming_set = set(incoming_ext_ids)

        # Для OpenSearch
        to_delete_os = list(set(os_extra))
        if to_delete_os:
            self.logger.warning(
                f"🗑 OpenSearch: удаляем {len(to_delete_os)} документов..."
            )
            try:
                deleted_count = self.os.delete_by_ext_ids(to_delete_os)
                self.logger.info(f"✅ OpenSearch: удалено {deleted_count}, ext_ids: {to_delete_os}")
            except Exception as e:
                self.logger.error(f"❌ Ошибка удаления в OpenSearch: {e}")

        # Для Milvus
        to_delete_milvus = list(set(mil_extra))
        if to_delete_milvus:
            self.logger.warning(
                f"🗑 Milvus: удаляем {len(to_delete_milvus)} entities..."
            )
            try:
                deleted_count = await self.milvus.delete_by_ext_ids(
                    self.collection_name, to_delete_milvus
                )
                self.logger.info(f"✅ Milvus: удалено ~{deleted_count}, ext_ids: {to_delete_milvus}")
            except Exception as e:
                self.logger.error(f"❌ Ошибка удаления в Milvus: {e}")

        # Определяем, что нужно обновлять/добавлять в OpenSearch
        new_in_os = set(os_missing)
        self.logger.info(
            f"🗑 OpenSearch: новых документов: {len(new_in_os)}, ext_ids: {new_in_os}"
        )
        update_candidates_os = set(os_found)

        # Для найденных в OS проверяем modified_at
        if update_candidates_os:
            self.logger.info("🔎 OpenSearch: сравниваем modified_at...")
            try:
                # Создаем карту modified_at только для найденных записей
                incoming_modified_map_os = {
                    str(r["ext_id"]): (
                        ""
                        if r.get("modified_at") is None
                        else str(r.get("modified_at")).strip()
                    )
                    for r in df[
                        df["ext_id"].astype(str).isin(update_candidates_os)
                    ].to_dict(orient="records")
                }

                os_different = set(
                    self.os.diff_modified_by_ext_ids(incoming_modified_map_os)
                )
            except Exception as e:
                self.logger.error(f"⚠️ Ошибка при diff_modified_by_ext_ids в OS: {e}")
                os_different = set()
        else:
            os_different = set()
        self.logger.info(
            f"🗑 OpenSearch: измененных документов: {len(os_different)}, ext_ids: {os_different}"
        )
        # Для Milvus аналогично
        new_in_milvus = set(mil_missing)
        self.logger.info(
            f"🗑 Milvus: новых документов: {len(new_in_milvus)}, ext_ids: {new_in_milvus}"
        )
        update_candidates_mil = set(mil_found)

        if update_candidates_mil:
            self.logger.info("🔎 Milvus: сравниваем modified_at...")
            try:
                # Создаем карту modified_at только для найденных записей
                incoming_modified_map_mil = {
                    str(r["ext_id"]): (
                        ""
                        if r.get("modified_at") is None
                        else str(r.get("modified_at")).strip()
                    )
                    for r in df[
                        df["ext_id"].astype(str).isin(update_candidates_mil)
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
            f"🗑 Milvus: измененных документов: {len(mil_different)}, ext_ids: {mil_different}"
        )
        # Определяем, что нужно создать/обновить в каждой БД отдельно
        to_upsert_os = (new_in_os | os_different) & incoming_set
        to_upsert_mil = (new_in_milvus | mil_different) & incoming_set

        # Объединяем для удобства обработки (уникальные записи)
        to_upsert_all = to_upsert_os | to_upsert_mil

        if not to_upsert_all:
            self.logger.info("✅ Нет новых или изменённых записей для upsert.")
            return

        df_to_upsert = df[df["ext_id"].astype(str).isin(to_upsert_all)].copy()
        if df_to_upsert.empty:
            return

        if self.model is None:
            self.model = SentenceTransformer(self.settings.milvus.model_name)

        docs = df_to_upsert[self.settings.milvus.search_fields].astype(str).tolist()
        metadata = df_to_upsert.to_dict(orient="records")

        self.logger.info(f"⬆️ Подготавливаем upsert для {len(metadata)} записей...")

        try:
            embeddings = await self.milvus.get_embeddings(self.model, docs)
        except Exception as e:
            self.logger.error(f"❌ Ошибка генерации эмбеддингов: {e}")
            raise

        # Upsert в Milvus (только для записей, которые нужно обновить в Milvus)
        mil_metadata = [m for m in metadata if str(m["ext_id"]) in to_upsert_mil]
        if mil_metadata:
            try:
                mil_indices = [
                    i
                    for i, m in enumerate(metadata)
                    if str(m["ext_id"]) in to_upsert_mil
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
        os_metadata = [m for m in metadata if str(m["ext_id"]) in to_upsert_os]
        if os_metadata:
            try:
                self.os.upsert(os_metadata)
                self.logger.info(
                    f"✅ OpenSearch: upsert выполнен для {len(os_metadata)} записей"
                )
            except Exception as e:
                self.logger.error(f"❌ OpenSearch upsert failed: {e}")

        self.logger.info("✅ Обновление коллекции завершено")

    async def update_vio_base(self) -> None:
        df = await self._load_excel_from_edu("vio")
        df = self._prepare_metadata(df)
        await self._update_collection_from_df(df, target_source="ВиО")
        cleanup_resources(self.logger)

    async def update_kb_base(self) -> None:
        df = await self._load_excel_from_edu("kb")
        df = self._prepare_metadata(df)
        await self._update_collection_from_df(df, target_source="ТП")
        cleanup_resources(self.logger)

    async def update_all(self) -> None:
        df_kb = await self._load_excel_from_edu("kb")
        df_kv = await self._load_excel_from_edu("vio")
        df_combined = pd.concat([df_kb, df_kv])

        df_renamed = rename_dataframe(df_combined)
        df_renamed = df_renamed.drop_duplicates(subset=['ext_id'], keep="last")
        df_deduped = dedup_by_question_any(df_renamed)
        df_deduped = self._prepare_metadata(df_deduped)

        df_kb = df_deduped[df_deduped["source"] == "ТП"].copy()
        df_kv = df_deduped[df_deduped["source"] == "ВиО"].copy()

        await self._update_collection_from_df(df_kb, target_source="ТП")
        await self._update_collection_from_df(df_kv, target_source="ВиО")

        cleanup_resources(self.logger)
