import asyncio
import typing as tp

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from pymilvus.exceptions import SchemaNotReadyException
from sentence_transformers import SentenceTransformer

from app.common.exceptions.exceptions import MilvusCollectionLoadTimeoutError, NotFoundError
from app.common.logger import AISearchLogger
from app.infrastructure.storages.interfaces import IVectorDatabase
from app.settings.config import MilvusSettings


class MilvusDatabase(IVectorDatabase):
    """Класс для работы с Milvus DB."""

    def __init__(self, settings: MilvusSettings, logger: AISearchLogger):
        self.config = settings
        self.logger = logger
        connections.connect("default", host=self.config.host, port=self.config.port)
        self.initialize_model_metadata_collection()

    @staticmethod
    def get_model_name(model: SentenceTransformer) -> str:
        """Получить имя модели"""
        return model._first_module().auto_model.config._name_or_path

    async def create_collection(self, collection_name: str, dim: int) -> None:
        """Создает коллекцию для хранения векторов."""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields, description=f"Collection {collection_name}")
        collection = Collection(collection_name, schema)
        await asyncio.to_thread(
            collection.create_index,
            "embedding",
            {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": self.config.nlist},
            },
            timeout=self.config.query_timeout,
        )

    async def insert_vectors(self, collection_name: str, vectors: list[list[float]]) -> None:
        """Вставка векторов и связанных метаданных в коллекцию."""
        collection = Collection(collection_name)
        entities = [
            list(range(len(vectors))),  # IDs
            vectors,  # Embeddings
        ]
        await asyncio.to_thread(collection.insert, entities, timeout=self.config.query_timeout)
        await asyncio.to_thread(collection.flush, timeout=self.config.query_timeout)

    async def search(
        self, collection_name: str, query_vector: list[float], top_k: int
    ) -> list[dict[str, tp.Any]]:
        """Поиск по косинусной схожести."""
        top_k = max(top_k, 1)
        collection: Collection = await self.load_collection_and_wait(
            collection_name=collection_name
        )
        search_params = {"metric_type": "COSINE", "params": {"nprobe": self.config.nprobe}}
        results: list[list[dict[str, tp.Any]]] = await asyncio.to_thread(
            collection.search,
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["id"],
            timeout=self.config.query_timeout,
        )
        return [{"id": hit["id"], "distance": hit["distance"]} for hits in results for hit in hits]

    async def collection_ready(self, collection_name: str) -> bool:
        """Проверка наличия колллекци."""
        if not await asyncio.to_thread(
            utility.has_collection, collection_name, timeout=self.config.query_timeout
        ):
            return False
        try:
            collection = Collection(collection_name)
            return await asyncio.to_thread(collection.has_index, timeout=self.config.query_timeout)
        except SchemaNotReadyException:
            return False

    async def delete_collection(self, collection_name: str) -> None:
        """Удаление коллекции."""
        collection = Collection(collection_name)
        await asyncio.to_thread(collection.drop, timeout=self.config.query_timeout)

    async def get_model_by_collection(self, collection_name: str) -> str:
        """Получение текущей модели из метаданных."""
        metadata_collection: Collection = await self.load_collection_and_wait(
            collection_name="model_metadata"
        )
        query = f"collection_name == '{collection_name}'"
        results = await asyncio.to_thread(
            metadata_collection.query,
            expr=query,
            output_fields=["model_name"],
            timeout=self.config.query_timeout,
        )
        if results:
            return results[0]["model_name"]
        else:
            raise NotFoundError

    async def update_model_metadata(self, collection_name: str, model_name: str) -> None:
        """Сохранение информации о модели в метаданные."""
        self.logger.info(
            f"Создание связи коллекция -> имя модели ({collection_name} -> {model_name})"
        )
        metadata_collection = Collection("model_metadata")
        entities = [
            [collection_name],
            [model_name],
            [[0.0, 0.0]],
        ]
        await asyncio.to_thread(
            metadata_collection.upsert, entities, timeout=self.config.query_timeout
        )
        await asyncio.to_thread(metadata_collection.flush, timeout=self.config.query_timeout)

    def initialize_model_metadata_collection(self) -> None:
        """Инициализация коллекции для метаданных модели."""
        if not utility.has_collection("model_metadata", timeout=self.config.query_timeout):
            fields = [
                FieldSchema(
                    name="collection_name", dtype=DataType.VARCHAR, max_length=255, is_primary=True
                ),
                FieldSchema(name="model_name", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(
                    name="dummy_vector", dtype=DataType.FLOAT_VECTOR, dim=2
                ),  # вектор-заглушка для хранения связи коллекция -> модель
            ]
            schema = CollectionSchema(fields, description="Метаданные моделей")
            collection = Collection("model_metadata", schema)
            collection.create_index(
                "dummy_vector",
                {
                    "metric_type": "L2",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1},
                },
                timeout=self.config.query_timeout,
            )
            self.logger.info("Коллекция 'model_metadata' успешно создана")

    async def load_collection_and_wait(self, collection_name: str) -> Collection:
        """Загрузка коллекции."""
        collection = Collection(collection_name)
        await asyncio.to_thread(collection.load, timeout=self.config.load_timeout)

        async def check_load_state() -> None:
            while not utility.load_state(collection_name):
                await asyncio.sleep(0.5)

        try:
            await asyncio.wait_for(check_load_state(), timeout=self.config.load_timeout)
        except asyncio.TimeoutError:
            raise MilvusCollectionLoadTimeoutError(
                f"Таймаут при загрузке коллекции '{collection_name}'."
            )
        return collection

    async def get_model_metadata(self, limit: int = 10) -> list[dict[str, tp.Any]]:
        """Получение результатов model_metadata"""
        metadata_collection: Collection = await self.load_collection_and_wait(
            collection_name="model_metadata"
        )
        results = await asyncio.to_thread(
            metadata_collection.query,
            expr="",
            limit=limit,
            output_fields=["model_name", "collection_name"],
        )
        return results

    async def index_documents(
        self, collection_name: str, model: SentenceTransformer, documents: list[str]
    ) -> None:
        """Индексация документов в vector_db."""
        embeddings = model.encode(documents)
        await self.create_collection(collection_name, dim=embeddings.shape[1])
        await self.insert_vectors(collection_name, embeddings.tolist())

    async def ensure_model_consistency(
        self, collection_name: str, model: SentenceTransformer, documents: list[str]
    ) -> bool:
        """Гарантирует актуальность связи коллекция -> модель
        Возвращает True, если модель была изменена
        """
        self.logger.info(await self.get_model_metadata())
        if not await self.collection_ready(collection_name):
            await self.initialize_collection(collection_name, model, documents)
            return False
        else:
            current_model = await self.get_model_by_collection(collection_name)
            if current_model != MilvusDatabase.get_model_name(model):
                await self.handle_model_change(collection_name, model, documents)
                return True
            return False

    async def initialize_collection(
        self, collection_name: str, model: SentenceTransformer, documents: list[str]
    ) -> None:
        """Инициализация коллекции с текущей моделью."""
        self.logger.info("Инициализация коллекции")
        await self.index_documents(collection_name, model, documents)
        await self.update_model_metadata(collection_name, MilvusDatabase.get_model_name(model))

    async def handle_model_change(
        self, collection_name: str, model: SentenceTransformer, documents: list[str]
    ) -> None:
        """Обработка смены модели."""
        old_model_name: str = await self.get_model_by_collection(collection_name=collection_name)
        self.logger.info(
            f"Смена модели: {old_model_name} -> {MilvusDatabase.get_model_name(model)}"
        )
        await self.delete_collection(collection_name)
        await self.initialize_collection(collection_name, model, documents)
