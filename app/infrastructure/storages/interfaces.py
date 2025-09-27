import abc
import typing as tp

import pandas as pd
from sentence_transformers import SentenceTransformer


class IVectorDatabase(abc.ABC):
    """Интерфейс для работы с векторными базами данных."""

    @staticmethod
    @abc.abstractmethod
    def get_model_name(model: SentenceTransformer) -> str:
        """Получить имя модели"""

    @abc.abstractmethod
    async def create_collection(self, collection_name: str, dim: int) -> None:
        """Создает коллекцию для хранения векторов."""
        pass

    @abc.abstractmethod
    async def insert_vectors(
        self, collection_name: str, vectors: list[list[float]], metadata: list[dict[str, tp.Any]]
    ) -> None:
        """Вставляет векторы и связанные метаданные в коллекцию."""
        pass

    @abc.abstractmethod
    async def search(
        self, collection_name: str, query_vector: list[float], top_k: int
    ) -> list[dict[str, tp.Any]]:
        """Выполняет поиск по косинусной схожести."""
        pass

    @abc.abstractmethod
    async def collection_ready(self, collection_name: str) -> bool:
        """Проверяет, существует ли коллекция."""
        pass

    @abc.abstractmethod
    async def delete_collection(self, collection_name: str) -> None:
        """Удаляет коллекцию."""
        pass

    @abc.abstractmethod
    async def get_model_by_collection(self, collection_name: str) -> str:
        """Получение текущей модели из метаданных."""
        pass

    @abc.abstractmethod
    async def update_model_metadata(self, collection_name: str, model_name: str) -> None:
        """Сохранение информации о модели в метаданные."""
        pass

    @abc.abstractmethod
    def initialize_model_metadata_collection(self) -> None:
        """Инициализация коллекции для метаданных модели."""

    @abc.abstractmethod
    def preload_collections(self) -> None:
        """Предзагрузка коллекций в память"""

    @abc.abstractmethod
    async def get_model_metadata(self, limit: int) -> list[dict[str, tp.Any]]:
        """Получение результатов model_metadata"""

    @abc.abstractmethod
    async def index_documents(
        self, collection_name: str, model: SentenceTransformer, documents: list[str]
    ) -> None:
        """Индексация документов в vector_db."""

    @abc.abstractmethod
    async def ensure_collection(
        self,
        collection_name: str,
        model: SentenceTransformer,
        documents: list[str],
        metadata: pd.DataFrame | None,
        recreate: bool,
    ) -> None:
        """Гарантирует готовность коллекции (актуальная модель) и пересоздаёт при необходимости"""

    @abc.abstractmethod
    async def initialize_collection(
        self, collection_name: str, model: SentenceTransformer, documents: list[str]
    ) -> None:
        """Инициализация коллекции с текущей моделью."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Закрытие соединения с клиентом."""
