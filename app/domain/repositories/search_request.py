from app.common.repositories.repository import SQLAlchemyRepository
from app.domain.repositories.interfaces import ISearchRequestRepository
from app.domain.schemas.search_request import SearchRequestSchema
from app.infrastructure.models.search_request import SearchRequest


class SearchRequestRepository(SQLAlchemyRepository, ISearchRequestRepository):
    """Репозиторий результатов поиска"""

    model = SearchRequest
    response_dto = SearchRequestSchema
