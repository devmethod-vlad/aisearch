import typing as tp
import time

import numpy as np

from pymilvus import (
    AsyncMilvusClient,
    CollectionSchema, DataType,
)
from sentence_transformers import SentenceTransformer

from app.common.logger import AISearchLogger
from app.infrastructure.storages.interfaces import IVectorDatabase
from app.infrastructure.utils.milvus import load_schema_and_indexes_from_json
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
        self._search_params_by_field = set()


        # Только предзагрузка коллекций (metadata логика удалена)

        metrics_print("🕒 Инициализация Milvus", milvus_init_start)

    @staticmethod
    def get_model_name(model: SentenceTransformer) -> str:
        """Получить имя модели"""
        return model._first_module().auto_model.config._name_or_path.split("/")[-1]

    async def create_collection(
            self,
            collection_name: str,
    ) -> None:
        """Создает коллекцию для хранения векторов"""
        collections = await self.client.list_collections(timeout=self.config.query_timeout)
        if collection_name in collections:
            await self.client.drop_collection(collection_name, timeout=self.config.query_timeout)

        fields, index_specs, search_params_by_field = load_schema_and_indexes_from_json(self.config.schema_path)

        schema = CollectionSchema(fields, description=f"Collection {collection_name}")
        await self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            timeout=self.config.query_timeout,
        )

        if index_specs:
            field_names = {f.name for f in fields}
            index_params = self.client.prepare_index_params()

            for idx in index_specs:
                if idx.field_name not in field_names:
                    raise ValueError(f"В JSON указан индекс для отсутствующего поля: {idx.field_name}")


                add_kwargs = {
                    "field_name": idx.field_name,
                    "index_type": idx.index_type,
                    "params": idx.params or {},
                }
                if idx.metric_type is not None:
                    add_kwargs["metric_type"] = idx.metric_type

                index_params.add_index(**add_kwargs)

            await self.client.create_index(
                collection_name=collection_name,
                index_params=index_params,
                timeout=self.config.query_timeout,
            )
        else:

            self.logger.warning(
            "🚨 Milvus: в JSON не задано ни одного индекса — коллекция будет работать в режиме FLAT (медленнее)."
            )


        self._search_params_by_field = search_params_by_field or {}

        await self.client.load_collection(collection_name, timeout=self.config.query_timeout)
        self.__collections_loaded.add(collection_name)

    async def insert_vectors(
            self,
            collection_name: str,
            vectors: list[list[float]],
            metadata: list[dict[str, tp.Any]] | None = None,
            batch_size: int = 512,
    ) -> None:
        """Вставка векторов и метаданных с проверкой типов и размерности."""
        vectors_size = len(vectors)

        if metadata is not None:
            data_size = len(metadata)
            if vectors_size != data_size:
                raise ValueError(
                    f"Количество векторов не равно количеству метаданных ({vectors_size} != {data_size})"
                )
        else:
            if vectors_size == 0:

                raise ValueError("Нельзя вставить 0 векторов без метаданных")
            metadata = [{} for _ in range(vectors_size)]

        fields, index_specs, search_params_by_field = load_schema_and_indexes_from_json(self.config.schema_path)
        f_by_name = {f.name: f for f in fields}

        vec_field = self.config.vector_field
        if vec_field not in f_by_name:
            raise ValueError(f"В схеме нет векторного поля '{vec_field}'")

        # dim может быть как атрибутом .dim, так и в params (в разных версиях pymilvus)
        vfs = f_by_name[vec_field]
        dim = getattr(vfs, "dim", None)
        if dim is None:
            params = getattr(vfs, "params", {}) or {}
            dim = params.get("dim")

        # 2) Нормализация одного вектора (тип и размерность)
        def _normalize_vector(vec: tp.Sequence[tp.Any]) -> list[float]:
            if dim is not None and len(vec) != dim:
                raise ValueError(f"Ожидалась размерность {dim}, получили {len(vec)}")
            try:
                return [float(x) for x in vec]
            except Exception as e:
                raise TypeError(f"Невозможно привести элементы вектора к float: {e}") from e


        def _coerce(name: str, value: tp.Any) -> tuple[tp.Any | None, bool]:
            """
            -> (coerced_value, drop)
            drop=True означает 'не включать это поле в запись'
            """
            f = f_by_name.get(name)
            if f is None:
                return None, True
            if value is None:
                return None, True

            dt = f.dtype
            if dt == DataType.VARCHAR:
                s = str(value)
                max_len = getattr(f, "max_length", None)
                if max_len:
                    s = s[:max_len]
                return s, False
            if dt in (DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64):
                return int(value), False
            if dt in (DataType.FLOAT, DataType.DOUBLE):
                return float(value), False
            if dt == DataType.BOOL:
                return bool(value), False
            if dt == DataType.FLOAT_VECTOR:

                return None, True

            return value, False


        num_batches = (vectors_size + batch_size - 1) // batch_size
        self.logger.info(
            f"Загрузка векторов{' и метаданных' if metadata else ''}, батч {batch_size}, всего батчей: {num_batches}"
        )

        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, vectors_size)

            vectors_batch = vectors[start:end]
            rows_batch = metadata[start:end]  # type: ignore[index]

            # Готовим данные построчно: {vector_field: [...], **coerced_meta}
            data: list[dict[str, tp.Any]] = []
            for vec, row in zip(vectors_batch, rows_batch):
                item: dict[str, tp.Any] = {}

                # вектор
                item[vec_field] = _normalize_vector(vec)

                # метаданные (только поля из схемы, без None)
                for k, v in row.items():
                    val, drop = _coerce(k, v)
                    if not drop:
                        item[k] = val

                data.append(item)

            # вставка батча
            await self.client.insert(
                collection_name=collection_name,
                data=data,
                timeout=self.config.query_timeout,
            )
            self.logger.info(f"Загружено {i + 1}/{num_batches} батчей")

        # 5) Финальный flush (один раз, после всех батчей)
        await self.client.flush(collection_name, timeout=self.config.query_timeout)

    async def search(
            self, collection_name: str, query_vector: list[float], top_k: int
    ) -> list[dict[str, tp.Any]]:
        """Поиск по косинусной схожести."""
        top_k = max(top_k, 1)

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

    async def collection_ready(self, collection_name: str) -> bool:
        """Проверка наличия и готовности коллекции."""
        try:
            collections = await self.client.list_collections(timeout=self.config.query_timeout)
            if collection_name not in collections:
                return False

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

    async def preload_collections(self) -> None:
        """Предзагрузка коллекций в память"""
        collection_name = self.config.collection_name
        try:
            self.logger.info(f"⏳ Загрузка коллекции {collection_name} ...")
            await self.client.load_collection(
                collection_name, timeout=self.config.query_timeout
            )
            self.__collections_loaded.add(collection_name)
            self.logger.info(f"✅ Коллекция {collection_name} успешно загружена")
        except Exception as e:
            self.logger.warning(f"⚠️ Не удалось загрузить коллекцию {collection_name}: {e}")

    async def index_documents(
            self,
            collection_name: str,
            model: SentenceTransformer,
            documents: list[str],
            metadata: list[dict[str, tp.Any]] | None = None,
    ) -> None:
        """Индексация документов в vector_db."""
        self.logger.info("ПРОИСХОДИТ ИНДЕКСАЦИЯ")
        embeddings = model.encode(documents, normalize_embeddings=True)
        embeddings = np.vstack([l2_normalize(e) for e in embeddings])

        await self.create_collection(
            collection_name
        )

        await self.insert_vectors(
            collection_name=collection_name, vectors=embeddings.tolist(), metadata=metadata
        )

    async def ensure_collection(
            self,
            collection_name: str,
            model: SentenceTransformer,
            documents: list[str] | None = None,
            metadata: list[dict[str, tp.Any]] | None = None,
            recreate: bool = False,
    ) -> None:
        """Гарантирует готовность коллекции."""

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

        if not await self.collection_ready(collection_name):
            self.logger.info(f"Коллекция {collection_name} не существует, создаём ...")
            if documents is None:
                raise ValueError("Для создания новой коллекции нужны documents")
            await self.initialize_collection(
                collection_name=collection_name, model=model, documents=documents, metadata=metadata
            )
            return

        if collection_name not in self.__collections_loaded:
            await self.client.load_collection(collection_name, timeout=self.config.query_timeout)
            self.__collections_loaded.add(collection_name)
            self.logger.info(f"Коллекция {collection_name} подгружена в память")

    async def initialize_collection(
            self,
            collection_name: str,
            model: SentenceTransformer,
            documents: list[str],
            metadata: list[dict[str, tp.Any]] | None = None,
    ) -> None:
        """Инициализация коллекции с текущей моделью."""
        self.logger.info(f"Инициализация коллекции {collection_name} ...")
        await self.index_documents(
            collection_name=collection_name, model=model, documents=documents, metadata=metadata
        )

    async def close(self) -> None:
        """Закрытие соединения с клиентом."""
        if hasattr(self, "client"):
            await self.client.close()
