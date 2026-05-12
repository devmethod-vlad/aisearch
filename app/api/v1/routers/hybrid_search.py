from dishka.integrations.fastapi import FromDishka, inject
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from app.api.v1.dto.requests.search import SearchRequest
from app.api.v1.dto.responses.taskmanager import TaskResponse
from app.common.fastapi_utils import TolerantAPIRouter
from app.services.interfaces import IHybridSearchService
from app.settings.config import settings

limiter = Limiter(key_func=get_remote_address)

router = TolerantAPIRouter(prefix="/hybrid-search", tags=["Hybrid Search"])


@router.post("/search", response_model=TaskResponse)
@limiter.limit(settings.slowapi.search, key_func=get_remote_address)
@inject
async def get_documents(
    request: Request,
    body: SearchRequest,
    service: FromDishka[IHybridSearchService] = None,
) -> JSONResponse:
    """Выполняет гибридный поиск. Ставит задачу в очередь и возвращает ticket_id."""
    query = body.query
    top_k = body.top_k
    filters = body.filters

    # Извлекаем оба типа фильтров в контракте внешнего API.
    array_filters = (
        filters.array_filters.model_dump(exclude_none=True)
        if filters and filters.array_filters
        else {}
    )
    exact_filters = (
        filters.exact_filters.model_dump(exclude_none=True)
        if filters and filters.exact_filters
        else {}
    )

    presearch = body.presearch.model_dump() if body.presearch else None

    response = await service.enqueue_search(
        query=query,
        top_k=top_k,
        array_filters=array_filters,
        exact_filters=exact_filters,
        search_use_cache=body.search_use_cache,
        show_intermediate_results=body.show_intermediate_results,
        metrics_enable=body.metrics_enable,
        presearch=presearch,
    )
    return JSONResponse(content=jsonable_encoder(response), status_code=202)




@router.get("/info/{ticket_id}", response_model=TaskResponse)
@inject
async def get_task_status(
    ticket_id: str,
    service: FromDishka[IHybridSearchService] = None,
) -> JSONResponse:
    """Проверяет статус задачи по ticket_id."""
    response = await service.get_task_status(ticket_id)
    return JSONResponse(
        content=jsonable_encoder(response, exclude_none=True), status_code=200
    )
