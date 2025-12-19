from abc import ABC

from app.common.repositories.interfaces import IRepository


class ISearchRequestRepository(IRepository, ABC):
    """Интерфейс репозитория результатов поиска"""

    pass


class ISearchFeedbackRepository(IRepository, ABC):
    """Интерфейс репозитория обратной связи по поиску"""

    pass


class IKnowledgeFeedbackRepository(IRepository, ABC):
    """Интерфейс репозитория оценки знания"""

    pass
