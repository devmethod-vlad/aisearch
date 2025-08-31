import typing as tp

from pydantic import BaseModel


class TaskQueryResponse(BaseModel):
    """Ответ для фронтенда по опросу статуса задачи"""

    task_id: str
    task_status_url: str


class TaskInfoResponse(BaseModel):
    """Информация о статусе задачи"""

    status: str
    info: tp.Any = None
