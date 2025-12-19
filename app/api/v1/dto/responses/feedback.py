import uuid

from app.api.v1.dto.responses.base import PaginatedResponse
from app.common.arbitrary_model import ArbitraryModel


class FeedbackCreateResponse(ArbitraryModel):
    """Ответ на создание фидбека"""

    status: str = "created"
    id: uuid.UUID


class FeedbackBulkCreateResponse(ArbitraryModel):
    """Ответ на массовое создание фидбеков"""

    status: str = "created"
    created_ids: list[uuid.UUID]


class SearchRequestsResponse(PaginatedResponse):
    """Ответ с search_request"""

    pass


class SearchFeedbacksResponse(PaginatedResponse):
    """Ответ с search_feedback"""

    pass


class KnowledgeFeedbacksResponse(PaginatedResponse):
    """Ответ с knowledge_feedback"""

    pass
