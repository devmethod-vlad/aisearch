from dishka.integrations.fastapi import FromDishka, inject
from fastapi import Path
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from app.api.v1.dto.requests.requests import SearchRequestQueryRequest
from app.api.v1.dto.responses.requests import SearchRequestsResponse
from app.common.arbitrary_model import ArbitraryModel
from app.common.fastapi_utils import TolerantAPIRouter
from app.services.interfaces import IRequestsService
from app.settings.config import settings

limiter = Limiter(key_func=get_remote_address)

router = TolerantAPIRouter(prefix="/requests", tags=["Requests"])


@router.post(
    "/list",
    response_model=SearchRequestsResponse,
    summary="Получение списка данных по запросам",
)
@limiter.limit(settings.slowapi.search, key_func=get_remote_address)
@inject
async def list_search_requests(
    request: Request,
    body: SearchRequestQueryRequest,
    service: FromDishka[IRequestsService] = None,
) -> JSONResponse:
    """Получение списка данных по запросам"""
    response = await service.get_search_requests(body)
    return JSONResponse(content=jsonable_encoder(response))


@router.get(
    "/{request_id}",
    response_model=ArbitraryModel,
    summary="Получение данных по конкретному запросу",
)
@limiter.limit(settings.slowapi.search, key_func=get_remote_address)
@inject
async def get_search_request(
    request: Request,
    request_id: str = Path(..., description="ID запроса"),
    service: FromDishka[IRequestsService] = None,
) -> JSONResponse:
    """Получение данных по запросу по его ID"""
    response = await service.get_search_request_by_id(request_id)
    return JSONResponse(content=jsonable_encoder(response))
