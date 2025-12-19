from app.common.repositories.repository import SQLAlchemyRepository
from app.domain.repositories.interfaces import ISearchFeedbackRepository
from app.domain.schemas.search_feedback import SearchFeedbackSchema
from app.infrastructure.models.search_feedback import SearchFeedback


class SearchFeedbackRepository(SQLAlchemyRepository, ISearchFeedbackRepository):
    """Репозиторий обратной связи по поиску"""

    model = SearchFeedback
    response_dto = SearchFeedbackSchema
