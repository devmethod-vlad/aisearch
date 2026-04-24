from __future__ import annotations

import time
import traceback
import typing as tp

import redis.asyncio as redis
from celery import shared_task
from celery.utils.log import get_task_logger

from app.common.logger import AISearchLogger, LoggerType
from app.infrastructure.search_worker.worker import (
    get_container_from_task,
    run_coroutine,
)
from app.services.interfaces import IHybridSearchOrchestrator
from app.settings.config import (
    settings,
)


@shared_task(name="search_task", bind=True, max_retries=3)
def search_task(
    self: tp.Callable, ticket_id: str, pack_key: str, result_key: str
) -> dict[str, tp.Any]:
    """Выполняет гибридный поиск в фоновом режиме."""
    container = get_container_from_task(search_task)

    async def run_task() -> dict[str, tp.Any]:
        redis_watcher = redis.from_url(str(settings.redis.dsn))
        tprefix = settings.llm_queue.ticket_hash_prefix
        pkey = (
            settings.llm_queue.processing_list_key
            or f"{settings.llm_queue.queue_list_key}:processing"
        )

        try:
            logger = await container.get(AISearchLogger)
            logger.info(
                f"Начало выполнения задачи 'search_task' для ticket_id: {ticket_id}"
            )

            task_logger_instance = get_task_logger(__name__)

            orchestrator = await container.get(IHybridSearchOrchestrator)
            orchestrator.set_logger(task_logger_instance)

            task_id = getattr(self, "task_id", None) or getattr(self.request, "id", "")
            result = await orchestrator.documents_search(
                task_id=task_id,
                ticket_id=ticket_id,
                pack_key=pack_key,
                result_key=result_key,
            )

            logger.info(f"Задача 'search_task' для ticket_id: {ticket_id} выполнена")
            return result

        except Exception as error:
            await redis_watcher.hset(
                f"{tprefix}{ticket_id}",
                mapping={
                    "state": "failed",
                    "error": f"({type(error)}) {traceback.format_exc()}",
                    "updated_at": int(time.time()),
                },
            )

            # Получаем логгер для логирования ошибки
            logger = AISearchLogger(logger_type=LoggerType.CELERY)
            logger.error(
                f"Ошибка в задаче поиска для ticket_id {ticket_id} ({type(error)}): {traceback.format_exc()}"
            )

            await redis_watcher.lrem(pkey, 1, ticket_id)
            return {"status": "failed"}

    return run_coroutine(run_task())
