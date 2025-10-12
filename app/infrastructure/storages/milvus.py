import typing as tp
import time

import numpy as np
import pandas as pd
from pymilvus import (
    AsyncMilvusClient,
    CollectionSchema,
    DataType,
    FieldSchema,
)
from sentence_transformers import SentenceTransformer

from app.common.exceptions.exceptions import (
    NotFoundError,
)
from app.common.logger import AISearchLogger
from app.infrastructure.storages.interfaces import IVectorDatabase
from app.infrastructure.utils.nlp import l2_normalize
from app.infrastructure.utils.metrics import metrics_print
from app.settings.config import MilvusSettings


class MilvusDatabase(IVectorDatabase):
    """Класс для работы с Milvus DB с использованием AsyncMilvusClient."""

    def __init__(self, settings: MilvusSettings, logger: AISearchLogger):
        milvus_init_start = time.perf_counter()
        self.config = settings
        self.logger = logger
        self.client = AsyncMilvusClient(
            uri=f"http{'s' if self.config.use_ssl else ''}://{self.config.host}:{self.config.port}",
            timeout=self.config.connection_timeout,
        )
        self.__collections_loaded = set()

        # Инициализация выполняется асинхронно
        self.initialize_model_metadata_collection()
        self.preload_collections()
        metrics_print("🕒 Инициализация Milvus", milvus_init_start)


    @staticmethod
    def get_model_name(model: SentenceTransformer) -> str:
        """Получить имя модели"""
        return model._first_module().auto_model.config._name_or_path.split("/")[-1]

    async def create_collection(
        self,
        collection_name: str,
        dim: int,
        metadata_fields: list[str],
        metadata_maxlen: int = 65535,
    ) -> None:
        """Создает коллекцию для хранения векторов"""
        # Проверяем, существует ли коллекция
        collections = await self.client.list_collections(timeout=self.config.query_timeout)
        if collection_name in collections:
            await self.client.drop_collection(collection_name, timeout=self.config.query_timeout)

        # Создаем схему полей
        fields = [
            FieldSchema(
                name=self.config.id_field, dtype=DataType.INT64, is_primary=True, auto_id=True
            ),
            FieldSchema(name=self.config.vector_field, dtype=DataType.FLOAT_VECTOR, dim=dim),
            *[
                FieldSchema(name=field, dtype=DataType.VARCHAR, max_length=metadata_maxlen)
                for field in metadata_fields
            ],
        ]

        schema = CollectionSchema(fields, description=f"Collection {collection_name}")

        # Создаем коллекцию
        await self.client.create_collection(
            collection_name=collection_name, schema=schema, timeout=self.config.query_timeout
        )

        # Создаем параметры индекса
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=self.config.vector_field,
            index_type="HNSW",
            metric_type=self.config.metric_type,
            params={"M": 16, "efConstruction": 128},
        )

        # Создаем индекс
        await self.client.create_index(
            collection_name=collection_name,
            index_params=index_params,
            timeout=self.config.query_timeout,
        )

        # Загружаем коллекцию
        await self.client.load_collection(collection_name, timeout=self.config.query_timeout)
        self.__collections_loaded.add(collection_name)

    async def insert_vectors(
        self,
        collection_name: str,
        vectors: list[list[float]],  # нормализованные векторы
        metadata: pd.DataFrame | None = None,
        batch_size: int = 512,
    ) -> None:
        """Вставка векторов и связанных метаданных в коллекцию."""
        vectors_size = len(vectors)

        if metadata is not None:
            data_size = len(metadata.index)
            if vectors_size != data_size:
                raise ValueError(
                    f"Количество векторов не равно количеству строк в DataFrame ({vectors_size} != {data_size})"
                )

            rows = []
            for _, r in metadata.iterrows():
                row_dict = {col: str(r[col]) for col in metadata.columns}
                rows.append(row_dict)
        else:
            rows = [{} for _ in range(vectors_size)]
            if vectors_size == 0:
                raise ValueError("Нельзя вставить 0 векторов без DataFrame")

        # Убеждаемся, что коллекция загружена
        if collection_name not in self.__collections_loaded:
            await self.client.load_collection(collection_name, timeout=self.config.query_timeout)
            self.__collections_loaded.add(collection_name)

        num_batches = (vectors_size + batch_size - 1) // batch_size
        self.logger.info(f"Загрузка векторов{' и метаданных' if metadata is not None else ''} ...")

        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, vectors_size)
            vectors_batch = vectors[start:end]
            rows_batch = rows[start:end]

            # Подготавливаем данные для вставки
            data = []
            for vector, row in zip(vectors_batch, rows_batch):
                item = {self.config.vector_field: vector, **row}
                data.append(item)

            # Вставляем данные
            await self.client.insert(
                collection_name=collection_name, data=data, timeout=self.config.query_timeout
            )
            self.logger.info(f"Загружено {i+1}/{num_batches} ...")

        # Выполняем flush для гарантии записи
        await self.client.flush(collection_name, timeout=self.config.query_timeout)

    async def search(
        self, collection_name: str, query_vector: list[float], top_k: int
    ) -> list[dict[str, tp.Any]]:
        """Поиск по косинусной схожести."""
        top_k = max(top_k, 1)

        # Убеждаемся, что коллекция загружена
        if collection_name not in self.__collections_loaded:
            await self.client.load_collection(collection_name, timeout=self.config.query_timeout)
            self.__collections_loaded.add(collection_name)

        results = await self.client.search(
            collection_name=collection_name,
            data=[query_vector],
            anns_field=self.config.vector_field,
            params={"metric_type": self.config.metric_type, "params": {"ef": 64}},
            limit=top_k,
            output_fields=self.config.output_fields,
            timeout=self.config.query_timeout,
        )

        out: list[dict[str, tp.Any]] = []
        if not results:
            return out
        hits = results[0]
        for h in hits:
            fields = h.entity
            row = {k: fields.get(k, "") for k in self.config.output_fields}
            row["score_dense"] = float(h.distance)
            out.append(row)
        return out

        # return [{"id": hit["id"], "distance": hit["distance"]} for hits in results for hit in hits]

    async def collection_ready(self, collection_name: str) -> bool:
        """Проверка наличия и готовности коллекции."""
        try:
            collections = await self.client.list_collections(timeout=self.config.query_timeout)
            if collection_name not in collections:
                return False

            # Проверяем, есть ли индекс
            indexes = await self.client.list_indexes(collection_name)
            return len(indexes) > 0

        except Exception:
            return False

    async def delete_collection(self, collection_name: str) -> None:
        """Удаление коллекции."""
        self.logger.info(f"Удаление коллекции {collection_name} ...")

        collections = await self.client.list_collections(timeout=self.config.query_timeout)
        if collection_name not in collections:
            self.logger.info(f"Коллекция {collection_name} не существует, удаление не требуется.")
            self.__collections_loaded.discard(collection_name)
            return

        await self.client.drop_collection(collection_name, timeout=self.config.query_timeout)
        self.__collections_loaded.discard(collection_name)
        self.logger.info(f"Коллекция {collection_name} успешно удалена")

    async def get_model_by_collection(self, collection_name: str) -> str:
        """Получение текущей модели из метаданных."""
        try:
            results = await self.client.query(
                collection_name="model_metadata",
                filter=f"collection_name == '{collection_name}'",
                output_fields=["model_name"],
                timeout=self.config.query_timeout,
            )

            if results:
                return results[0]["model_name"]
            else:
                raise NotFoundError(f"Модель для коллекции {collection_name} не найдена")
        except Exception as e:
            raise NotFoundError(f"Ошибка при получении модели: {e}")

    async def update_model_metadata(self, collection_name: str, model_name: str) -> None:
        """Сохранение информации о модели в метаданные."""
        self.logger.info(
            f"Создание связи коллекция -> имя модели ({collection_name} -> {model_name}) ..."
        )

        data = [
            {
                "collection_name": collection_name,
                "model_name": model_name,
                "dummy_vector": [0.0, 0.0],  # вектор-заглушка
            }
        ]

        await self.client.upsert(
            collection_name="model_metadata", data=data, timeout=self.config.query_timeout
        )
        await self.client.flush("model_metadata", timeout=self.config.query_timeout)

    async def initialize_model_metadata_collection(self) -> None:
        """Инициализация коллекции для метаданных модели."""
        collections = await self.client.list_collections(timeout=self.config.query_timeout)

        if "model_metadata" not in collections:
            fields = [
                FieldSchema(
                    name="collection_name", dtype=DataType.VARCHAR, max_length=255, is_primary=True
                ),
                FieldSchema(name="model_name", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="dummy_vector", dtype=DataType.FLOAT_VECTOR, dim=2),
            ]

            schema = CollectionSchema(fields, description="Метаданные моделей")

            await self.client.create_collection(
                collection_name="model_metadata", schema=schema, timeout=self.config.query_timeout
            )

            # Создаем параметры индекса для model_metadata
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="dummy_vector",
                index_type="IVF_FLAT",
                metric_type="L2",
                params={"nlist": 1},
            )

            await self.client.create_index(
                collection_name="model_metadata",
                index_params=index_params,
                timeout=self.config.query_timeout,
            )

            self.logger.info("✅ Коллекция 'model_metadata' успешно создана")

        # Загружаем коллекцию
        await self.client.load_collection("model_metadata", timeout=self.config.query_timeout)
        self.__collections_loaded.add("model_metadata")

    async def preload_collections(self) -> None:
        """Предзагрузка коллекций в память"""
        for collection_name in self.config.preloaded_collection_names:
            try:
                self.logger.info(f"⏳ Загрузка коллекции {collection_name} ...")
                await self.client.load_collection(
                    collection_name, timeout=self.config.query_timeout
                )
                self.__collections_loaded.add(collection_name)
                self.logger.info(f"✅ Коллекция {collection_name} успешно загружена")
            except Exception as e:
                self.logger.warning(f"⚠️ Не удалось загрузить коллекцию {collection_name}: {e}")

    async def get_model_metadata(self, limit: int = 10) -> list[dict[str, tp.Any]]:
        """Получение результатов model_metadata"""
        try:
            results = await self.client.query(
                collection_name="model_metadata",
                filter="",
                limit=limit,
                output_fields=["model_name", "collection_name"],
                timeout=self.config.query_timeout,
            )
            return results
        except Exception as e:
            self.logger.error(f"⚠️ Ошибка при получении метаданных моделей: {e}")
            return []

    async def index_documents(
        self,
        collection_name: str,
        model: SentenceTransformer,
        documents: list[str],
        metadata: pd.DataFrame | None = None,
    ) -> None:
        """Индексация документов в vector_db."""
        self.logger.info("ПРОИСХОДИТ ИНДЕКСАЦИЯ")
        embeddings = model.encode(documents, normalize_embeddings=True)
        embeddings = np.vstack([l2_normalize(e) for e in embeddings])

        metadata_fields = metadata.columns.tolist() if metadata is not None else []
        await self.create_collection(
            collection_name,
            dim=embeddings.shape[1],
            metadata_fields=metadata_fields,
        )

        await self.insert_vectors(
            collection_name=collection_name, vectors=embeddings.tolist(), metadata=metadata
        )

    async def ensure_collection(
            self,
            collection_name: str,
            model: SentenceTransformer,
            documents: list[str] | None = None,
            metadata: pd.DataFrame | None = None,
            recreate: bool = False,
    ) -> None:
        """Гарантирует готовность коллекции.

        Поведение:
        - recreate=True → пересоздание коллекции с документами
        - если коллекция не существует → создаётся и индексируется
        - если коллекция есть → просто load_collection
        - если модель не совпадает → логируем warning, но НЕ пересоздаём
        """

        if recreate:
            self.logger.info(f"⏳ Пересоздание коллекции {collection_name} (recreate=True) ...")
            if await self.collection_ready(collection_name):
                await self.delete_collection(collection_name=collection_name)
            if documents is None:
                raise ValueError("Для recreate=True нужно передать documents")
            await self.initialize_collection(
                collection_name=collection_name, model=model, documents=documents, metadata=metadata
            )
            return

        # Проверяем существование
        if not await self.collection_ready(collection_name):
            self.logger.info(f"Коллекция {collection_name} не существует, создаём ...")
            if documents is None:
                raise ValueError("Для создания новой коллекции нужны documents")
            await self.initialize_collection(
                collection_name=collection_name, model=model, documents=documents, metadata=metadata
            )
            return

        # Коллекция существует → просто load_collection
        if collection_name not in self.__collections_loaded:
            await self.client.load_collection(collection_name, timeout=self.config.query_timeout)
            self.__collections_loaded.add(collection_name)
            self.logger.info(f"Коллекция {collection_name} подгружена в память")

        # Проверка модели (но без пересоздания!)
        try:
            current_model_name = await self.get_model_by_collection(collection_name)
            expected_model_name = MilvusDatabase.get_model_name(model)
            if current_model_name != expected_model_name:
                self.logger.warning(
                    f"Коллекция {collection_name} создана с моделью {current_model_name}, "
                    f"а текущая модель {expected_model_name}. "
                    f"Поиск будет выполняться, но результаты могут быть некорректны."
                )
        except NotFoundError:
            self.logger.warning(
                f"Коллекция {collection_name} существует, но модель не найдена в метаданных. "
                f"Рекомендуется пересоздать коллекцию вручную."
            )

        # Модель успешно получена, проверяем, совпадает ли она
        expected_model_name = MilvusDatabase.get_model_name(model)
        if current_model_name != expected_model_name:
            self.logger.info(
                f"Модель изменилась: {current_model_name} -> {expected_model_name}. Пересоздаём коллекцию."
            )
            await self.delete_collection(collection_name=collection_name)
            await self.initialize_collection(
                collection_name=collection_name, model=model, documents=documents, metadata=metadata
            )

    async def initialize_collection(
        self,
        collection_name: str,
        model: SentenceTransformer,
        documents: list[str],
        metadata: pd.DataFrame | None = None,
    ) -> None:
        """Инициализация коллекции с текущей моделью."""
        self.logger.info(f"Инициализация коллекции {collection_name} ...")
        await self.index_documents(
            collection_name=collection_name, model=model, documents=documents, metadata=metadata
        )
        await self.update_model_metadata(collection_name, MilvusDatabase.get_model_name(model))

    async def close(self) -> None:
        """Закрытие соединения с клиентом."""
        if hasattr(self, "client"):
            await self.client.close()
