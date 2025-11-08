import abc
import typing as tp

from sentence_transformers import SentenceTransformer

class IVectorDatabase(abc.ABC):
    """Интерфейс для работы с векторными базами данных."""

    @staticmethod
    @abc.abstractmethod
    def get_model_name(model: SentenceTransformer) -> str:
        """Получить имя модели"""

    @abc.abstractmethod
    async def create_collection(
        self,
        collection_name: str,
    ) -> None:
        """Создает коллекцию для хранения векторов."""

    @abc.abstractmethod
    async def insert_vectors(
        self,
        collection_name: str,
        vectors: list[list[float]],
        metadata: list[dict[str, tp.Any]] | None = None,
        batch_size: int = 512,
    ) -> None:
        """Вставляет векторы и связанные метаданные в коллекцию."""

    @abc.abstractmethod
    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[dict[str, tp.Any]]:
        """Выполняет поиск по косинусной схожести."""

    @abc.abstractmethod
    async def collection_ready(self, collection_name: str) -> bool:
        """Проверяет, существует ли коллекция и готова ли к работе."""

    @abc.abstractmethod
    async def delete_collection(self, collection_name: str) -> None:
        """Удаляет коллекцию."""

    @abc.abstractmethod
    async def preload_collections(self) -> None:
        """Предзагрузка коллекций в память."""

    @abc.abstractmethod
    async def index_documents(
        self,
        collection_name: str,
        model: SentenceTransformer,
        documents: list[str],
        metadata: list[dict[str, tp.Any]] | None = None,
    ) -> None:
        """Индексация документов в vector_db."""

    @abc.abstractmethod
    async def ensure_collection(
        self,
        collection_name: str,
        model: SentenceTransformer,
        documents: list[str] | None = None,
        metadata: list[dict[str, tp.Any]] | None = None,
        recreate: bool = False,
    ) -> None:
        """Гарантирует готовность коллекции (создаёт или пересоздаёт при необходимости)."""

    @abc.abstractmethod
    async def initialize_collection(
        self,
        collection_name: str,
        model: SentenceTransformer,
        documents: list[str],
        metadata: list[dict[str, tp.Any]] | None = None,
    ) -> None:
        """Инициализация коллекции с текущей моделью."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Закрытие соединения с клиентом."""

    @abc.abstractmethod
    async def fetch_existing(
            self,
            collection_name: str,
            output_fields: list[str] | None = None,
    ) -> list[dict[str, tp.Any]]:
        """Извлечь существующие данные"""

    @abc.abstractmethod
    async def upsert_vectors(
            self,
            collection_name: str,
            vectors: list[list[float]],
            metadata: list[dict[str, tp.Any]] | None = None,
            batch_size: int = 512,
    ):
        """Вставка/обновление векторов и метаданных (upsert по ext_id)."""

    @abc.abstractmethod
    async def delete_vectors(
            self,
            collection_name: str,
            ext_ids: list[str] | None = None,
            filter_expr: str | None = None,
    ) -> None:
        """Удаление записей"""
