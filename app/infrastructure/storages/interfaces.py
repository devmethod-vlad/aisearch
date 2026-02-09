import abc
import typing as tp

import numpy as np
from sentence_transformers import SentenceTransformer


class IVectorDatabase(abc.ABC):
    """Адаптер для векторной базы данных"""

    @abc.abstractmethod
    async def load_collection(self, collection_name: str) -> None:
        """Подгрузка коллекции"""

    @abc.abstractmethod
    async def create_collection(self, collection_name: str) -> None:
        """Создает коллекцию для хранения векторов"""

    @abc.abstractmethod
    async def insert_vectors(
        self,
        collection_name: str,
        vectors: list[list[float]],
        metadata: list[dict[str, tp.Any]] | None = None,
        batch_size: int = 512,
    ) -> None:
        """Вставка векторов и метаданных с проверкой типов и размерности."""

    @abc.abstractmethod
    async def search(
        self, collection_name: str, query_vector: list[float], top_k: int
    ) -> list[dict[str, tp.Any]]:
        """Поиск по косинусной схожести."""

    @abc.abstractmethod
    async def collection_ready(self, collection_name: str) -> bool:
        """Проверка наличия и готовности коллекции."""

    @abc.abstractmethod
    async def delete_collection(self, collection_name: str) -> None:
        """Удаление коллекции."""

    @abc.abstractmethod
    async def preload_collections(self) -> None:
        """Предзагрузка коллекций в память"""

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
    async def initialize_collection(
        self,
        collection_name: str,
        model: SentenceTransformer,
        documents: list[str],
        metadata: list[dict[str, tp.Any]] | None = None,
    ) -> None:
        """Инициализация коллекции с текущей моделью."""

    @abc.abstractmethod
    async def get_embeddings(
        self, model: SentenceTransformer, documents: list[str]
    ) -> np.ndarray:
        """Получение эмбеддингов"""

    @abc.abstractmethod
    async def close(self) -> None:
        """Закрытие соединения с клиентом."""

    @abc.abstractmethod
    async def fetch_existing(
        self, collection_name: str, output_fields: list[str] | None = None
    ) -> list[dict]:
        """Получить все данные из коллекции"""

    @abc.abstractmethod
    async def upsert_vectors(
        self,
        collection_name: str,
        vectors: list[list[float]],
        metadata: list[dict[str, tp.Any]] | None = None,
        batch_size: int = 512,
    ) -> None:
        """Вставка/обновление векторов и метаданных (upsert по ext_id)."""

    @abc.abstractmethod
    async def delete_vectors(
        self,
        collection_name: str,
        ext_ids: list[str] | None = None,
        filter_expr: str | None = None,
    ) -> None:
        """Удаляет записи из коллекции Milvus."""

    @abc.abstractmethod
    async def collection_not_empty(self, collection_name: str) -> bool:
        """Проверка, что коллекция существует и содержит хотя бы 1 запись"""

    @abc.abstractmethod
    async def find_existing_ext_ids(
        self,
        collection_name: str,
        incoming_ext_ids: tp.Iterable[tp.Any],
        field: str = "ext_id",
        batch_size: int = 1000,
        iterator_batch: int = 4096,
        source_field: str | None = None,
        source: str | None = None,
    ) -> tuple[list[str], list[str], list[str]]:
        """Возвращает:
        - found_incoming: входящие ext_id, найденные в коллекции
        - missing_incoming: входящие ext_id, которых нет в коллекции
        - extra_in_store: ext_id из коллекции, отсутствующие во входящих
        """

    @abc.abstractmethod
    async def delete_by_ext_ids(
        self,
        collection_name: str,
        ext_ids: list[str],
        field: str = "ext_id",
        batch_size: int = 1000,
    ) -> int:
        """Удаляет entities по строковому полю (VarChar) батчами.
        Возвращает количество удалённых entities.
        """

    @abc.abstractmethod
    async def diff_modified_by_ext_ids(
        self,
        collection_name: str,
        incoming_modified: dict[str, str],
        *,
        field: str = "ext_id",
        modified_field: str = "modified_at",
        batch_size: int = 1000,
    ) -> list[str]:
        """Возвращает список ext_id, у которых modified_at в Milvus отличается
        от входящего значения.
        """
