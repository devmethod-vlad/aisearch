import gc
import importlib
import io
import typing as tp
import numpy as np
import pandas as pd
import torch
import unicodedata
from sentence_transformers import SentenceTransformer

from app.common.logger import AISearchLogger
from app.infrastructure.adapters.interfaces import IOpenSearchAdapter, IEduAdapter
from app.infrastructure.storages.interfaces import IVectorDatabase
from app.infrastructure.utils.nlp import l2_normalize
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
        "Обновлено": "modified_at"
    }

    def __init__(self, settings: Settings, logger: AISearchLogger,
                 edu: IEduAdapter, milvus: IVectorDatabase, os: IOpenSearchAdapter):
        self.settings = settings
        self.logger = logger
        self.edu = edu
        self.milvus = milvus
        self.os_adapter = os
        self.collection_name = settings.milvus.collection_name
        self.model: tp.Optional[SentenceTransformer] = None

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

    def _prepare_metadata(self, df: pd.DataFrame, file_type: str) -> pd.DataFrame:
        df = df.copy()
        df.rename(columns=self.FIELD_MAPPING, inplace=True)
        df["ext_id"] = df["ext_id"].astype(str)
        df = df[df["answer"].astype(str).str.len() > 2]
        df["row_idx"] = range(len(df))
        if file_type == "vio":
            df["space"] = df["space"].astype(str).str.strip()

            df = df[
                df["space"].notna()
                & (df["space"].str.strip() != "")
                & (df["space"].str.lower() != "не распределено")
                ]

        return df

    async def _fetch_existing_data(self) -> dict[str, dict]:
        """Собираем все существующие записи из Milvus и OS по ext_id"""
        all_fields = list(UpdaterService.FIELD_MAPPING.values()) + [self.settings.milvus.vector_field, "row_idx"]
        milvus_raw = await self.milvus.fetch_existing(self.collection_name, output_fields=all_fields)
        milvus_data = {str(r["ext_id"]): r for r in milvus_raw if r.get("ext_id")}
        os_raw = self.os_adapter.fetch_existing()
        os_data = {str(r["ext_id"]): r for r in os_raw if r.get("ext_id")}
        combined = milvus_data.copy()
        combined.update(os_data)  # OS перезаписывает пересечения
        return combined

    def normalize_text(self, val):
        """Приводим значение к строке, убираем невидимые символы, нормализуем переносы строк."""
        if val is None:
            return None
        if isinstance(val, float) and np.isnan(val):
            return None
        if isinstance(val, str) and (val.strip() == "" or val.strip().lower() == "nan"):
            return None

        s = str(val).strip()
        s = s.replace("\xa0", " ")  # неразрывные пробелы
        s = s.replace("\r\n", "\n").replace("\r", "\n")  # нормализуем CRLF и CR
        # Заменяем подряд идущие переносы на один
        s = "\n".join([line.strip() for line in s.splitlines() if line.strip() != ""])
        s = unicodedata.normalize("NFKC", s)
        return s

    def _diff_records(self, incoming_df: pd.DataFrame, existing_data: dict[str, dict]) -> pd.DataFrame:
        """
        Возвращает только новые или изменённые строки.
        Если есть поле modified_at — сравниваем только его.
        Если его нет или пустое — сравниваем все поля.
        """
        to_update = []

        for _, row in incoming_df.iterrows():
            ext_id = str(row["ext_id"])
            existing_row = existing_data.get(ext_id)


            if not existing_row:
                to_update.append(row)
                continue

            val_incoming_mod = str(row.get("modified_at") or "").strip()
            val_existing_mod = str(existing_row.get("modified_at") or "").strip()

            # Если есть modified_at, сравниваем только его
            if val_incoming_mod and val_existing_mod:
                if val_incoming_mod != val_existing_mod:
                    self.logger.warning(f"🕓 Изменено {ext_id}: modified_at {val_existing_mod!r} -> {val_incoming_mod!r}")
                    to_update.append(row)
                continue

            # Если modified_at нет — сравниваем всё остальное
            for col in incoming_df.columns:
                if col in ("row_idx", "modified_at"):
                    continue

                val_incoming = self.normalize_text(row[col])
                val_existing = self.normalize_text(existing_row.get(col))

                if val_incoming != val_existing:
                    self.logger.warning(f"✏️ Изменено {ext_id}: {col} — {val_existing!r} -> {val_incoming!r}")
                    to_update.append(row)
                    break

        return pd.DataFrame(to_update)

    async def _update_collection_from_df(self, df: pd.DataFrame):
        self.logger.info("🔍 Сравнение с текущими данными ...")
        existing_data = await self._fetch_existing_data()

        # --- 1️⃣ Сравнение и определение изменённых ---
        df_to_update = self._diff_records(df, existing_data)

        # --- 2️⃣ Определяем, какие ext_id больше не актуальны ---
        incoming_ids = set(df["ext_id"].astype(str))
        existing_ids = set(existing_data.keys())
        to_delete_ids = existing_ids - incoming_ids

        # --- 3️⃣ Логирование ---
        if to_delete_ids:
            self.logger.warning(f"🗑 Найдено {len(to_delete_ids)} устаревших записей для удаления")
        if df_to_update.empty and not to_delete_ids:
            self.logger.info("✅ Нет новых, изменённых или удалённых записей — обновление не требуется.")
            return

        # --- 4️⃣ Upsert новых/изменённых ---
        if not df_to_update.empty:
            if self.model is None:
                self.model = SentenceTransformer(self.settings.milvus.model_name)

            documents = df_to_update[self.settings.milvus.search_fields].astype(str).tolist()
            metadata = df_to_update.to_dict(orient="records")

            self.logger.info(f"⬆️ Добавляем/обновляем {len(df_to_update)} записей ...")

            embeddings = self.model.encode(documents, normalize_embeddings=True)
            embeddings = np.vstack([l2_normalize(e) for e in embeddings])
            await self.milvus.upsert_vectors(self.collection_name, embeddings.tolist(), metadata)

            self.os_adapter.upsert(metadata)

        # --- 5️⃣ Удаление устаревших ---
        if to_delete_ids:
            try:
                await self.milvus.delete_vectors(self.collection_name, list(to_delete_ids))
                self.os_adapter.delete(list(to_delete_ids))
                self.logger.info(f"✅ Удалено {len(to_delete_ids)} записей из Milvus и OpenSearch")
            except Exception as e:
                self.logger.error(f"❌ Ошибка при удалении устаревших записей: {e}")

        self.logger.info("✅ Обновление базы завершено")

    async def update_vio_base(self):
        harvest = await self.edu.provoke_harvest_to_edu(harvest_type="vio")
        if harvest:
            df = await self._load_excel_from_edu("vio")
            df = self._prepare_metadata(df, "vio")
            await self._update_collection_from_df(df)
            await self.cleanup_resources()
        else:
            self.logger.error("Не удалось загрузить данные на edu")

    async def update_kb_base(self):
        harvest = await self.edu.provoke_harvest_to_edu(harvest_type="kb")
        if harvest:
            df = await self._load_excel_from_edu("kb")
            df = self._prepare_metadata(df, "kb")
            await self._update_collection_from_df(df)
            await self.cleanup_resources()
        else:
            self.logger.error("Не удалось загрузить данные на edu")

    async def cleanup_resources(self):
        self.logger.info("🧹 Очистка ресурсов ...")
        importlib.invalidate_caches()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Ресурсы очищены ✅")