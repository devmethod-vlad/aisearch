import json

from celery.result import AsyncResult

from app.api.v1.dto.responses.taskmanager import TaskInfoResponse
from app.common.logger import AISearchLogger
from app.common.storages.interfaces import KeyValueStorageProtocol
from app.services.interfaces import ITaskManagerService
from app.settings.config import RestrictionSettings


class TaskManagerService(ITaskManagerService):
    """Сервис таск-менеджер"""

    def __init__(
        self,
        logger: AISearchLogger,
        restriction_settings: RestrictionSettings,
        redis_storage: KeyValueStorageProtocol,
    ):
        self.logger = logger
        self.restriction_settings = restriction_settings
        self.redis_storage = redis_storage

    async def get_task_info(self, task_id: str, cache_key: str) -> TaskInfoResponse:
        """Получение информации о задаче"""
        task_result = AsyncResult(str(task_id))
        if task_result.state == "SUCCESS":
            result = task_result.result
            # dto_result = [
            #     ExampleResponse(
            #         id=r["id"], document=get_documents()[r["id"]], distance=r["distance"]
            #     )
            #     for r in result
            # ]
            await self.redis_storage.set(
                f"{self.restriction_settings.base_cache_key}:{cache_key}",
                json.dumps(result),
                ttl=self.restriction_settings.max_cache_ttl,
            )
            return TaskInfoResponse(status=task_result.state, info=task_result.result)
        else:
            return TaskInfoResponse(status=task_result.state)
