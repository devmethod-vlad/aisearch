import uuid

from app.api.v1.dto.requests.requests import SearchRequestQueryRequest
from app.api.v1.dto.responses.requests import SearchRequestsResponse
from app.common.arbitrary_model import ArbitraryModel
from app.common.exceptions.exceptions import NotFoundError
from app.common.filters.filters import (
    BaseFilter,
    Condition,
    DateFilter,
    OrderDirection,
    OrderingFilter,
    PaginationFilter,
    UUIDFilter,
)
from app.common.logger import AISearchLogger
from app.domain.filters.search_request import SearchRequestFilter
from app.infrastructure.unit_of_work.interfaces import IUnitOfWork
from app.services.interfaces import IRequestsService


class RequestsService(IRequestsService):
    """Сервис запросов"""

    def __init__(self, uow: IUnitOfWork, logger: AISearchLogger):
        self.uow: IUnitOfWork = uow
        self.logger = logger

    async def get_search_requests(
        self, request: SearchRequestQueryRequest
    ) -> SearchRequestsResponse:
        """Получение search_request с пагинацией и фильтрацией по времени"""
        async with self.uow:

            items, total, has_more = await self.uow.search_request.get_paginated(
                filters=BaseFilter(
                    condition=Condition.AND,
                    nested_filters=(
                        [
                            SearchRequestFilter(
                                modified_at=DateFilter(ge=request.start_time)
                            )
                        ]
                        if request.start_time
                        else None
                    ),
                    ordering=[
                        OrderingFilter(
                            field="modified_at", direction=OrderDirection.DESC
                        )
                    ],
                    pagination=PaginationFilter(
                        limit=request.limit, offset=request.offset
                    ),
                ),
            )

            return SearchRequestsResponse(
                items=items,
                total=total,
                limit=request.limit,
                offset=request.offset,
                has_more=has_more,
            )

    async def get_search_request_by_id(self, request_id: str) -> ArbitraryModel:
        """Получение search_request по ID"""
        async with self.uow:
            try:
                request_uuid = uuid.UUID(request_id)
            except ValueError:
                raise NotFoundError

            search_request = await self.uow.search_request.get_one(
                filters=SearchRequestFilter(id=UUIDFilter(eq=request_uuid))
            )
            return search_request
