import json
from datetime import UTC, datetime, timedelta

import shortuuid
from celery.result import AsyncResult

from app.api.v1.dto.responses.semantic_search import ExampleResponse
from app.api.v1.dto.responses.taskmanager import (
    TaskQueryResponse,
)
from app.common.logger import AISearchLogger
from app.common.storages.interfaces import KeyValueStorageProtocol
from app.domain.exceptions import QueueMaxSizeException, TimeoutException
from app.infrastructure.storages.interfaces import IVectorDatabase
from app.infrastructure.utils.nlp import get_documents, hash_query, normalize_query
from app.infrastructure.worker.tasks.semantic_search import task_semantic_search
from app.services.interfaces import ISemanticSearchService
from app.settings.config import AppSettings, RestrictionSettings


class SemanticSearchService(ISemanticSearchService):
    """Сервис семантического поиска."""

    def __init__(
        self,
        app_config: AppSettings,
        vector_db: IVectorDatabase,
        logger: AISearchLogger,
        redis_storage: KeyValueStorageProtocol,
        restrictions_settings: RestrictionSettings,
    ):
        self.config = app_config
        self.vector_db = vector_db
        self.logger = logger
        self.redis_storage = redis_storage
        self.restrictions_settings = restrictions_settings

    async def search(
        self, query: str, collection_name: str, top_k: int
    ) -> TaskQueryResponse | list[ExampleResponse]:
        """Семантический поиск"""
        top_k = max(top_k, 1)

        await self._update_last_query_time()
        await self._validate_queue_size()

        normalized_query = normalize_query(query)
        query_key = hash_query(query)

        cached = await self.redis_storage.get(
            f"{self.restrictions_settings.base_cache_key}:{query_key}"
        )
        if cached:
            cached_result = [
                ExampleResponse(
                    id=item["id"],
                    document=get_documents()[item["id"]],
                    distance=item["distance"],
                )
                for item in json.loads(cached)
            ]
            if len(cached_result) >= top_k:
                return cached_result[:top_k]

        task: AsyncResult = task_semantic_search.apply_async(
            args=[collection_name, normalized_query, query_key, top_k],
            task_id=f"semantic-search:{shortuuid.uuid()}",
        )

        return TaskQueryResponse(
            task_id=task.task_id,
            task_status_url=f"/taskmanager/{task.task_id}/{query_key}",
        )

    async def _validate_queue_size(self) -> int:
        """Проверка текущего размера очереди задач Celery"""
        current_tasks_ids = (
            await self.redis_storage.list_range(
                key=self.restrictions_settings.queue_key, start=0, end=-1
            )
            or []
        )
        semantic_search_tasks_ids = list(
            filter(lambda task_id: str(task_id).startswith("semantic-search"), current_tasks_ids)
        )
        amount_of_tasks = len(semantic_search_tasks_ids)
        if amount_of_tasks >= self.restrictions_settings.semantic_search_queue_size:
            raise QueueMaxSizeException("Очередь переполнена")
        return amount_of_tasks

    async def _update_last_query_time(self) -> None:
        """Обновление времени последнего запроса"""
        last_time = await self.redis_storage.get(
            self.restrictions_settings.semantic_search_last_query_time_key
        )
        if last_time:
            last_dt = datetime.fromisoformat(last_time)
            if datetime.now(UTC) - last_dt < timedelta(
                seconds=self.restrictions_settings.semantic_search_timeout_interval
            ):
                raise TimeoutException(
                    f"Большое количество запросов за последние "
                    f"{self.restrictions_settings.semantic_search_timeout_interval} секунд(ы)"
                )

        await self.redis_storage.set(
            self.restrictions_settings.semantic_search_last_query_time_key,
            datetime.now(UTC).isoformat(),
        )
