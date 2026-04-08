from pydantic import Field

from app.api.v1.dto.requests.base import PaginatedRequest, TimeIntervalRequest


class SearchRequestQueryRequest(PaginatedRequest, TimeIntervalRequest):
    """Запрос для получения search_request"""

    limit: int = Field(default=100, ge=1, le=200)
