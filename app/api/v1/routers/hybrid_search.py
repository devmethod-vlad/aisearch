from dishka.integrations.fastapi import FromDishka, inject
from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from app.api.v1.dto.responses.taskmanager import TaskResponse
from app.services.interfaces import IHybridSearchService

router = APIRouter(prefix="/hybrid-search", tags=["Hybrid Search"])


@router.post("/search", response_model=TaskResponse)
@inject
async def get_documents(
    query: str,
    top_k: int = 5,
    service: FromDishka[IHybridSearchService] = None,
) -> JSONResponse:
    """Выполняет гибридный поиск. Ставит задачу в очередь и возвращает ticket_id."""
    response = await service.enqueue_search(query=query, top_k=top_k)
    return JSONResponse(content=jsonable_encoder(response), status_code=202)


@router.post("/generate", response_model=TaskResponse)
@inject
async def generate_answer(
    query: str,
    top_k: int = 5,
    system_prompt: str | None = None,
    service: FromDishka[IHybridSearchService] = None,
) -> JSONResponse:
    """Генерирует ответ. Ставит задачу в очередь и возвращает ticket_id."""
    response = await service.enqueue_generate(query=query, top_k=top_k, system_prompt=system_prompt)
    return JSONResponse(content=jsonable_encoder(response, exclude_none=True), status_code=202)


@router.get("/info/{ticket_id}", response_model=TaskResponse)
@inject
async def get_task_status(
    ticket_id: str,
    service: FromDishka[IHybridSearchService] = None,
) -> JSONResponse:
    """Проверяет статус задачи по ticket_id."""
    response = await service.get_task_status(ticket_id)
    return JSONResponse(content=jsonable_encoder(response, exclude_none=True), status_code=200)
