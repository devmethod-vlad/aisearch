from dishka import FromDishka
from dishka.integrations.fastapi import inject
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from app.api.v1.dto.responses.taskmanager import TaskResponse
from app.common.fastapi_utils import TolerantAPIRouter
from app.services.interfaces import ITaskManagerService

router = TolerantAPIRouter(prefix="/taskmanager", tags=["Task Manager"])


@router.get(
    "/{task_id}",
    response_model=TaskResponse,
    summary="Получение информации о задаче",
)
@inject
async def task_info(
    service: FromDishka[ITaskManagerService], task_id: str
) -> JSONResponse:
    """Получение информации о задаче"""
    response: TaskResponse = await service.get_task_info(
        task_id=task_id,
    )
    return JSONResponse(
        content=jsonable_encoder(response, exclude_none=True), status_code=200
    )
