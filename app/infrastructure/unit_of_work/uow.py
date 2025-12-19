from app.common.uow.base_uow import BaseUnitOfWork
from app.domain.repositories.knowledge_feedback import KnowledgeFeedbackRepository
from app.domain.repositories.search_feedback import SearchFeedbackRepository
from app.domain.repositories.search_request import SearchRequestRepository
from app.infrastructure.unit_of_work.interfaces import IUnitOfWork


class UnitOfWork(BaseUnitOfWork, IUnitOfWork):
    """Единица работы"""

    async def __aenter__(self) -> None:
        """Инициализация сессии и репозиториев."""
        await super().__aenter__()
        self.search_request = SearchRequestRepository(session=self.session)
        self.search_feedback = SearchFeedbackRepository(session=self.session)
        self.knowledge_feedback = KnowledgeFeedbackRepository(session=self.session)
