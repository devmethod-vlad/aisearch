import uuid
from datetime import UTC, datetime

from app.api.v1.dto.requests.feedback import (
    KnowledgeFeedbackBulkCreateRequest,
    KnowledgeFeedbackCreateRequest,
    KnowledgeFeedbackQueryRequest,
    SearchFeedbackBulkCreateRequest,
    SearchFeedbackCreateRequest,
    SearchFeedbackQueryRequest,
    SearchRequestQueryRequest,
)
from app.api.v1.dto.responses.feedback import (
    FeedbackBulkCreateResponse,
    FeedbackCreateResponse,
    KnowledgeFeedbacksResponse,
    SearchFeedbacksResponse,
    SearchRequestsResponse,
)
from app.common.exceptions.exceptions import ConflictError
from app.common.filters.filters import (
    BaseFilter,
    Condition,
    DateFilter,
    OrderDirection,
    OrderingFilter,
    PaginationFilter,
)
from app.common.logger import AISearchLogger
from app.domain.exceptions import FeedbackException
from app.domain.filters.knowledge_feedback import KnowledgeFeedbackFilter
from app.domain.filters.search_feedback import SearchFeedbackFilter
from app.domain.filters.search_request import SearchRequestFilter
from app.domain.schemas.knowledge_feedback import KnowledgeFeedbackCreateDTO
from app.domain.schemas.search_feedback import SearchFeedbackCreateDTO
from app.infrastructure.unit_of_work.interfaces import IUnitOfWork
from app.services.interfaces import IFeedbackService


class FeedbackService(IFeedbackService):
    """Сервис сбора обратной связи"""

    def __init__(self, uow: IUnitOfWork, logger: AISearchLogger):
        self.uow: IUnitOfWork = uow
        self.logger = logger

    async def create_search_feedback(
        self, request: SearchFeedbackCreateRequest
    ) -> FeedbackCreateResponse:
        """Создание обратной связи по результату поиска"""
        async with self.uow:
            create_dto = SearchFeedbackCreateDTO(
                id=uuid.uuid4(),
                search_request_id=request.search_request_id,
                question_id=request.question_id,
                rating=request.rating,
                source=request.source,
                comment=request.comment,
                created_at=datetime.now(UTC),
                modified_at=datetime.now(UTC),
            )

            try:
                created_feedback = await self.uow.search_feedback.create(
                    create_dto=create_dto
                )
            except ConflictError:
                raise FeedbackException

            await self.uow.commit()

            self.logger.info(
                f"Search feedback created: {created_feedback.id}, "
                f"search_request_id: {request.search_request_id}, "
                f"question_id: {request.question_id}"
            )

            return FeedbackCreateResponse(id=created_feedback.id)

    async def create_knowledge_feedback(
        self, request: KnowledgeFeedbackCreateRequest
    ) -> FeedbackCreateResponse:
        """Создание оценки знания"""
        async with self.uow:
            create_dto = KnowledgeFeedbackCreateDTO(
                id=uuid.uuid4(),
                search_request_id=request.search_request_id,
                question_id=request.question_id,
                rating=request.rating,
                source=request.source,
                comment=request.comment,
                created_at=datetime.now(UTC),
                modified_at=datetime.now(UTC),
            )

            try:
                created_feedback = await self.uow.knowledge_feedback.create(
                    create_dto=create_dto
                )
            except ConflictError:
                raise FeedbackException

            await self.uow.commit()

            self.logger.info(
                f"Knowledge feedback created: {created_feedback.id}, "
                f"search_request_id: {request.search_request_id}, "
                f"question_id: {request.question_id}"
            )

            return FeedbackCreateResponse(id=created_feedback.id)

    async def bulk_create_search_feedback(
        self, request: SearchFeedbackBulkCreateRequest
    ) -> FeedbackBulkCreateResponse:
        """Массовое создание обратной связи по поиску"""
        async with self.uow:
            create_dtos = [
                SearchFeedbackCreateDTO(
                    id=uuid.uuid4(),
                    search_request_id=feedback.search_request_id,
                    question_id=feedback.question_id,
                    rating=feedback.rating,
                    source=feedback.source,
                    comment=feedback.comment,
                    created_at=datetime.now(UTC),
                    modified_at=datetime.now(UTC),
                )
                for feedback in request.feedbacks
            ]

            try:
                created_feedbacks = await self.uow.search_feedback.bulk_create(
                    bulk_create_dto=create_dtos
                )
            except ConflictError:
                raise FeedbackException

            await self.uow.commit()

            created_ids = [feedback.id for feedback in created_feedbacks]

            self.logger.info(
                f"Bulk created {len(created_ids)} search feedbacks: {created_ids}"
            )

            return FeedbackBulkCreateResponse(created_ids=created_ids)

    async def bulk_create_knowledge_feedback(
        self, request: KnowledgeFeedbackBulkCreateRequest
    ) -> FeedbackBulkCreateResponse:
        """Массовое создание оценки знания"""
        async with self.uow:
            create_dtos = [
                KnowledgeFeedbackCreateDTO(
                    id=uuid.uuid4(),
                    search_request_id=feedback.search_request_id,
                    question_id=feedback.question_id,
                    rating=feedback.rating,
                    source=feedback.source,
                    comment=feedback.comment,
                    created_at=datetime.now(UTC),
                    modified_at=datetime.now(UTC),
                )
                for feedback in request.feedbacks
            ]

            try:
                created_feedbacks = await self.uow.knowledge_feedback.bulk_create(
                    bulk_create_dto=create_dtos
                )
            except ConflictError:
                raise FeedbackException

            await self.uow.commit()

            created_ids = [feedback.id for feedback in created_feedbacks]

            self.logger.info(
                f"Bulk created {len(created_ids)} knowledge feedbacks: {created_ids}"
            )

            return FeedbackBulkCreateResponse(created_ids=created_ids)

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

    async def get_search_feedbacks(
        self, request: SearchFeedbackQueryRequest
    ) -> SearchFeedbacksResponse:
        """Получение search_feedback с пагинацией и фильтрацией по времени"""
        async with self.uow:
            items, total, has_more = await self.uow.search_feedback.get_paginated(
                filters=BaseFilter(
                    condition=Condition.AND,
                    nested_filters=(
                        [
                            SearchFeedbackFilter(
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

            return SearchFeedbacksResponse(
                items=items,
                total=total,
                limit=request.limit,
                offset=request.offset,
                has_more=has_more,
            )

    async def get_knowledge_feedbacks(
        self, request: KnowledgeFeedbackQueryRequest
    ) -> KnowledgeFeedbacksResponse:
        """Получение knowledge_feedback с пагинацией и фильтрацией по времени"""
        async with self.uow:
            items, total, has_more = await self.uow.knowledge_feedback.get_paginated(
                filters=BaseFilter(
                    condition=Condition.AND,
                    nested_filters=(
                        [
                            KnowledgeFeedbackFilter(
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

            return KnowledgeFeedbacksResponse(
                items=items,
                total=total,
                limit=request.limit,
                offset=request.offset,
                has_more=has_more,
            )
