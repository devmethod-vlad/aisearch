import abc
import typing as tp
from typing import Optional


from app.api.v1.dto.responses.taskmanager import TaskResponse


class ITaskManagerService(abc.ABC):
    """Интерфейс сервиса таск-менеджер"""

    @abc.abstractmethod
    async def get_task_info(self, task_id: str, cache_key: str) -> TaskResponse:
        """Получение информации о задаче"""


class IHybridSearchService(abc.ABC):
    """Сервис гибридного поиска"""

    @abc.abstractmethod
    async def enqueue_search(self, query: str, top_k: int) -> TaskResponse:
        """постановка задачи поиска по документам в очередь"""
        pass

    @abc.abstractmethod
    async def enqueue_generate(
        self, query: str, top_k: int, system_prompt: Optional[str]
    ) -> TaskResponse:
        """Постановка задачи генерации в очередь"""
        pass

    @abc.abstractmethod
    async def get_task_status(self, ticket_id: str) -> TaskResponse:
        """Получение статус задачи"""
        pass


class IHybridSearchOrchestrator(abc.ABC):
    """Оркестратор гибридного поиска"""

    @abc.abstractmethod
    async def documents_search(
        self,
        task_id: str,
        ticket_id: str,
        pack_key: str,
        result_key: str,
    ) -> dict[str, tp.Any]:
        """Поиск по документам"""

    @abc.abstractmethod
    async def warmup(self) -> bool:
        """Прогрев"""
