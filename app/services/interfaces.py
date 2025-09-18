import abc
from typing import Optional

from app.api.v1.dto.responses.semantic_search import ExampleResponse
from app.api.v1.dto.responses.taskmanager import (
    TaskInfoResponse,
    TaskQueryResponse,
)


class ISemanticSearchService(abc.ABC):
    """Интерфейс сервиса семантического поиска."""

    @abc.abstractmethod
    async def search(
        self, query: str, collection_name: str, top_k: int
    ) -> TaskQueryResponse | list[ExampleResponse]:
        """Семантический поиск"""


class ITaskManagerService(abc.ABC):
    """Интерфейс сервиса таск-менеджер"""

    @abc.abstractmethod
    async def get_task_info(self, task_id: str, cache_key: str) -> TaskInfoResponse:
        """Получение информации о задаче"""

    # @abc.abstractmethod
    # async def get_task_result(self, task_id: str, query_key: str) -> TaskInfoResponse:
    #     """Получение результатов поиска"""


class IHybridSearchService(abc.ABC):
    """Сервис гибридного поиска"""

    @abc.abstractmethod
    async def enqueue_search(self, query: str, top_k: int) -> TaskQueryResponse:
        """постановка задачи поиска по документам в очередь"""
        pass

    @abc.abstractmethod
    async def enqueue_generate(
        self, query: str, top_k: int, system_prompt: Optional[str]
    ) -> TaskQueryResponse:
        """Постановка задачи генерации в очередь"""
        pass

    @abc.abstractmethod
    async def get_task_status(self, ticket_id: str) -> TaskQueryResponse:
        """Получение статус задачи"""
        pass
