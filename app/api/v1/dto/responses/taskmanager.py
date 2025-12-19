import typing as tp

from pydantic import BaseModel


class TaskResponse(BaseModel):
    """Информация о задаче"""

    task_id: str
    url: str | None = None
    status: str | None = None
    extra: dict | None = None
    answer: str | None = None
    info: tp.Any = None
