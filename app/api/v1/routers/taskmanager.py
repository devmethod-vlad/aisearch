from dishka import FromDishka
from dishka.integrations.fastapi import inject
from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from app.api.v1.dto.responses.taskmanager import TaskInfoResponse
from app.services.interfaces import ITaskManagerService

router = APIRouter(prefix="/taskmanager", tags=["Task Manager"])


@router.get(
    "/{task_id}/{cache_key}",
    response_model=TaskInfoResponse,
    summary="Получение информации о задаче",
)
@inject
async def task_info(
    service: FromDishka[ITaskManagerService], task_id: str, cache_key: str
) -> JSONResponse:
    """Обогащение пользователя ролью"""
    response: TaskInfoResponse = await service.get_task_info(task_id=task_id, cache_key=cache_key)
    return JSONResponse(content=jsonable_encoder(response, exclude_none=True), status_code=200)
