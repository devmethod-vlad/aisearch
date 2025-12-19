from celery.result import AsyncResult

from app.api.v1.dto.responses.taskmanager import TaskResponse
from app.common.logger import AISearchLogger
from app.common.storages.interfaces import KeyValueStorageProtocol
from app.services.interfaces import ITaskManagerService
from app.settings.config import RedisSettings


class TaskManagerService(ITaskManagerService):
    """Сервис таск-менеджер"""

    def __init__(
        self,
        logger: AISearchLogger,
        redis_settings: RedisSettings,
        redis_storage: KeyValueStorageProtocol,
    ):
        self.logger = logger
        self.redis_settings = redis_settings
        self.redis_storage = redis_storage

    async def get_task_info(self, task_id: str) -> TaskResponse:
        """Получение информации о задаче"""
        task_result = AsyncResult(str(task_id))
        return TaskResponse(
            task_id=task_id,
            url=f"/taskmanager/{task_id}",
            status=task_result.state,
            info=task_result.result,
        )
