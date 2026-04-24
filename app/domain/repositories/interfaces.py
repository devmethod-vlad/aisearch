import abc
from abc import ABC

from app.common.repositories.interfaces import IRepository
from app.domain.schemas.glossary import GlossaryElementCreateDTO


class ISearchRequestRepository(IRepository, ABC):
    """Интерфейс репозитория результатов поиска"""

    pass


class ISearchFeedbackRepository(IRepository, ABC):
    """Интерфейс репозитория обратной связи по поиску"""

    pass


class IKnowledgeFeedbackRepository(IRepository, ABC):
    """Интерфейс репозитория оценки знания"""

    pass


class IGlossaryRepository(IRepository, abc.ABC):
    """Репозиторий элементов глоссария."""

    @abc.abstractmethod
    async def replace_all(
        self, elements: list[GlossaryElementCreateDTO], batch_size: int = 5000
    ) -> int:
        """Полностью заменяет содержимое таблицы glossary_element и обновляет materialized view."""
