import abc
import logging
import typing as tp
from abc import ABC, abstractmethod

from app.api.v1.dto.requests.feedback import (
    KnowledgeFeedbackBulkCreateRequest,
    KnowledgeFeedbackCreateRequest,
    KnowledgeFeedbackQueryRequest,
    SearchFeedbackBulkCreateRequest,
    SearchFeedbackCreateRequest,
    SearchFeedbackQueryRequest,
)
from app.api.v1.dto.requests.requests import SearchRequestQueryRequest
from app.api.v1.dto.responses.feedback import (
    FeedbackBulkCreateResponse,
    FeedbackCreateResponse,
    KnowledgeFeedbacksResponse,
    SearchFeedbacksResponse,
)
from app.api.v1.dto.responses.requests import SearchRequestsResponse
from app.api.v1.dto.responses.taskmanager import TaskResponse
from app.common.arbitrary_model import ArbitraryModel


class ITaskManagerService(abc.ABC):
    """Интерфейс сервиса таск-менеджер"""

    @abc.abstractmethod
    async def get_task_info(self, task_id: str, cache_key: str) -> TaskResponse:
        """Получение информации о задаче"""


class IHybridSearchService(abc.ABC):
    """Сервис гибридного поиска"""

    @abc.abstractmethod
    async def enqueue_search(
        self,
        query: str,
        top_k: int,
        array_filters: dict[str, list[str] | None] | None = None,
        exact_filters: dict[str, str | None] | None = None,
    ) -> TaskResponse:
        """Постановка задачи поиска с metadata-фильтрами.

        `array_filters` содержит фильтры по мультизначным полям
        `role`/`product`/`component` (внешний API-контракт).
        `exact_filters` содержит фильтры по single-value полям
        `source`/`actual`/`second_line`.

        Дальнейшая нормализация и применение выполняются в worker/orchestrator.
        """
        pass


    @abc.abstractmethod
    async def get_task_status(self, ticket_id: str) -> TaskResponse:
        """Получение статус задачи"""
        pass


class IHybridSearchOrchestrator(abc.ABC):
    """Оркестратор гибридного поиска"""

    @abc.abstractmethod
    def set_logger(self, logger: logging.Logger) -> None:
        """Переопределение логгера"""

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


class IUpdaterService(abc.ABC):
    """Обновляет бд из внешних источников"""

    @abc.abstractmethod
    async def update_vio_base(self) -> None:
        """Обновление данных из VIO"""

    @abc.abstractmethod
    async def update_kb_base(self) -> None:
        """Обновление данных из KB"""

    @abc.abstractmethod
    async def update_all(self) -> None:
        """Обновление всех источников"""


class IGlossaryService(abc.ABC):
    """Сервис синхронизации глоссария."""

    @abc.abstractmethod
    async def sync_glossary(self) -> int:
        """Получает глоссарий из внешнего API и полностью заменяет данные в БД."""


class IFeedbackService(ABC):
    """Интерфейс сервиса обратной связи"""

    @abstractmethod
    async def create_search_feedback(
        self, request: SearchFeedbackCreateRequest
    ) -> FeedbackCreateResponse:
        """Создание обратной связи по результату поиска"""

    @abstractmethod
    async def create_knowledge_feedback(
        self, request: KnowledgeFeedbackCreateRequest
    ) -> FeedbackCreateResponse:
        """Создание оценки знания"""

    @abstractmethod
    async def bulk_create_search_feedback(
        self, request: SearchFeedbackBulkCreateRequest
    ) -> FeedbackBulkCreateResponse:
        """Массовое создание обратной связи по поиску"""

    @abstractmethod
    async def bulk_create_knowledge_feedback(
        self, request: KnowledgeFeedbackBulkCreateRequest
    ) -> FeedbackBulkCreateResponse:
        """Массовое создание оценки знания"""

    @abstractmethod
    async def get_search_feedbacks(
        self, request: SearchFeedbackQueryRequest
    ) -> SearchFeedbacksResponse:
        """Получение search_feedback с пагинацией и фильтрацией по времени"""

    @abstractmethod
    async def get_knowledge_feedbacks(
        self, request: KnowledgeFeedbackQueryRequest
    ) -> KnowledgeFeedbacksResponse:
        """Получение knowledge_feedback с пагинацией и фильтрацией по времени"""


class IRequestsService(ABC):
    """Интерфейс сервиса запросов"""

    @abstractmethod
    async def get_search_requests(
        self, request: SearchRequestQueryRequest
    ) -> SearchRequestsResponse:
        """Получение search_request с пагинацией и фильтрацией по времени"""

    @abstractmethod
    async def get_search_request_by_id(self, request_id: str) -> ArbitraryModel:
        """Получение search_request по ID"""
