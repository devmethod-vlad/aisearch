import abc

from app.common.uow.interfaces import BaseAbstractUnitOfWork
from app.domain.repositories.interfaces import (
    IKnowledgeFeedbackRepository,
    ISearchFeedbackRepository,
    ISearchRequestRepository,
)


class IUnitOfWork(BaseAbstractUnitOfWork, abc.ABC):
    """Интерфейс единицы работы."""

    search_request: ISearchRequestRepository
    search_feedback: ISearchFeedbackRepository
    knowledge_feedback: IKnowledgeFeedbackRepository
