import typing as tp

from pydantic import BaseModel


class TaskResponse(BaseModel):
    """Информация о задаче"""

    task_id: str
    url: str
    status: tp.Optional[str] = None
    extra: tp.Optional[dict] = None
    answer: tp.Optional[str] = None
    info: tp.Any = None
