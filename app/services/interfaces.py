import abc

from app.api.v1.dto.responses.knowledge_base import CollectDataResponse
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


class IKnowledgeBaseService(abc.ABC):
    """Интерфейс сервиса базы знаний"""

    @abc.abstractmethod
    async def collect_data_from_confluence_by_page_id(self, page_id: str) -> CollectDataResponse:
        """Сбор данных БЗ из страницы Confluence"""

    @abc.abstractmethod
    async def collect_data_from_confluence_by_page_id_detached(
        self, page_id: str
    ) -> TaskQueryResponse:
        """Сбор данных БЗ из страницы Confluence в фоновом режиме"""

    @abc.abstractmethod
    async def collect_all_data_from_confluence(self) -> CollectDataResponse:
        """Сбор всех данных БЗ"""

    @abc.abstractmethod
    async def collect_all_data_from_confluence_detached(self) -> TaskQueryResponse:
        """Сбор всех данных БЗ в фоновом режиме"""
