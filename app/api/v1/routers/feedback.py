from dishka.integrations.fastapi import FromDishka, inject
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from app.api.v1.dto.requests.feedback import (
    KnowledgeFeedbackBulkCreateRequest,
    KnowledgeFeedbackCreateRequest,
    SearchFeedbackBulkCreateRequest,
    SearchFeedbackCreateRequest,
    SearchFeedbackQueryRequest,
    SearchRequestQueryRequest,
)
from app.api.v1.dto.responses.feedback import (
    FeedbackBulkCreateResponse,
    FeedbackCreateResponse,
    SearchFeedbacksResponse,
    SearchRequestsResponse,
)
from app.common.fastapi_utils import TolerantAPIRouter
from app.services.interfaces import IFeedbackService
from app.settings.config import settings

limiter = Limiter(key_func=get_remote_address)

router = TolerantAPIRouter(prefix="/feedback", tags=["Feedback"])


@router.post(
    "/search",
    response_model=FeedbackCreateResponse,
    summary="Создание обратной связи по поиску",
)
@inject
async def create_search_feedback(
    feedback: SearchFeedbackCreateRequest,
    service: FromDishka[IFeedbackService],
) -> JSONResponse:
    """Создание обратной связи по поиску"""
    result = await service.create_search_feedback(request=feedback)
    return JSONResponse(content=jsonable_encoder(result), status_code=201)


@router.post(
    "/knowledge",
    response_model=FeedbackCreateResponse,
    summary="Создание оценки знания",
)
@inject
async def create_knowledge_feedback(
    feedback: KnowledgeFeedbackCreateRequest,
    service: FromDishka[IFeedbackService],
) -> JSONResponse:
    """Создание оценки знания"""
    result = await service.create_knowledge_feedback(request=feedback)
    return JSONResponse(content=jsonable_encoder(result), status_code=201)


@router.post(
    "/search/bulk",
    response_model=FeedbackBulkCreateResponse,
    summary="Массовое создание обратной связи по поиску",
)
@inject
async def bulk_create_search_feedback(
    feedbacks: list[SearchFeedbackCreateRequest],
    service: FromDishka[IFeedbackService],
) -> JSONResponse:
    """Массовое создание обратной связи по поиску"""
    result = await service.bulk_create_search_feedback(
        request=SearchFeedbackBulkCreateRequest(feedbacks=feedbacks)
    )
    return JSONResponse(content=jsonable_encoder(result), status_code=201)


@router.post(
    "/knowledge/bulk",
    response_model=FeedbackBulkCreateResponse,
    summary="Массовое создание оценки знания",
)
@inject
async def bulk_create_knowledge_feedback(
    feedbacks: list[KnowledgeFeedbackCreateRequest],
    service: FromDishka[IFeedbackService],
) -> JSONResponse:
    """Массовое создание оценки знания"""
    result = await service.bulk_create_knowledge_feedback(
        request=KnowledgeFeedbackBulkCreateRequest(feedbacks=feedbacks)
    )
    return JSONResponse(content=jsonable_encoder(result), status_code=201)


@router.post(
    "/request/list",
    response_model=SearchRequestsResponse,
    summary="Получение списка данных по запросам",
)
@limiter.limit(settings.slowapi.search, key_func=get_remote_address)
@inject
async def list_search_requests(
    request: Request,
    body: SearchRequestQueryRequest,
    service: FromDishka[IFeedbackService] = None,
) -> JSONResponse:
    """Получение списка данных по запросам"""
    response = await service.get_search_requests(body)
    return JSONResponse(content=jsonable_encoder(response))


@router.post(
    "/search/list",
    response_model=SearchFeedbacksResponse,
    summary="Получение списка обратной связи по поиску",
)
@limiter.limit(settings.slowapi.search, key_func=get_remote_address)
@inject
async def list_search_feedbacks(
    request: Request,
    body: SearchFeedbackQueryRequest,
    service: FromDishka[IFeedbackService] = None,
) -> JSONResponse:
    """Получение списка обратной связи по поиску"""
    response = await service.get_search_feedbacks(body)
    return JSONResponse(content=jsonable_encoder(response))


@router.post(
    "/knowledge/list",
    response_model=SearchFeedbacksResponse,
    summary="Получение списка оценок знания",
)
@limiter.limit(settings.slowapi.search, key_func=get_remote_address)
@inject
async def list_knowledge_feedbacks(
    request: Request,
    body: SearchFeedbackQueryRequest,
    service: FromDishka[IFeedbackService] = None,
) -> JSONResponse:
    """Получение списка оценок знания"""
    response = await service.get_knowledge_feedbacks(body)
    return JSONResponse(content=jsonable_encoder(response))
