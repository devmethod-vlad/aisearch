from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import typing as tp
from asyncio import Future
from typing import TYPE_CHECKING

from celery import Celery
from celery.signals import (
    task_prerun,
    worker_process_init,
    worker_process_shutdown,
)

if TYPE_CHECKING:
    from dishka import AsyncContainer

from dishka import AsyncContainer, make_async_container

from app.common.logger import AISearchLogger, LoggerType
from app.common.storages.interfaces import KeyValueStorageProtocol
from app.infrastructure.adapters.interfaces import ILLMQueue, IVLLMAdapter
from app.infrastructure.utils.nlp import download_nltk_resources
from app.settings.config import HybridSearchSettings, Settings, settings

container: AsyncContainer | None = None
model: tp.Any | None = None
drain_task: asyncio.Task | None = None
logger: AISearchLogger | None = None

worker = Celery(
    "aisearch",
    broker=str(settings.redis.dsn),
    backend=str(settings.redis.dsn),
    include=["app.infrastructure.worker.tasks"],
)

worker.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    task_queues={"default": {"exchange": "default", "routing_key": "default"}},
    task_routes={"search_task": {"queue": "default"}, "generate_task": {"queue": "default"}},
    result_expires=600,
)

worker.autodiscover_tasks(["app.infrastructure.worker.tasks"])

# --- Единый event loop ---
loop = asyncio.new_event_loop()


@worker.on_after_configure.connect
def on_after_configure(**kwargs: dict[str, tp.Any]) -> None:
    """Код, выполняющийся после инициализации Celery и его настроек"""
    global logger  # noqa: PLW0603
    logger = AISearchLogger(logger_type=LoggerType.CELERY)
    download_nltk_resources()


def init_container_and_model() -> AsyncContainer:
    """Инициализация контейнера Dishka"""
    global container, model  # noqa: PLW0603
    from app.infrastructure.ioc import ApplicationProvider
    from app.infrastructure.providers import (
        LoggerProvider,
        MilvusProvider,
        RedisProvider,
    )
    from app.settings.config import (
        AppSettings,
        LLMGlobalSemaphoreSettings,
        LLMQueueSettings,
        MilvusSettings,
        RedisSettings,
        VLLMSettings,
    )

    container = make_async_container(
        ApplicationProvider(),
        LoggerProvider(),
        MilvusProvider(),
        RedisProvider(),
        context={
            AppSettings: settings.app,
            Settings: settings,
            MilvusSettings: settings.milvus,
            RedisSettings: settings.redis,
            LoggerType: LoggerType.CELERY,
            HybridSearchSettings: settings.hybrid,
            LLMGlobalSemaphoreSettings: settings.llm_global_sem,
            LLMQueueSettings: settings.llm_queue,
            VLLMSettings: settings.vllm,
        },
    )

    logger.info(f"Выполняется загрузка модели {settings.milvus.model_name.split('/')[-1]} ...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.milvus.model_name)
    logger.info("Модель успешно загружена")
    return container


def run_coroutine(coro: tp.Coroutine) -> Future:
    """Запуск корутин в общем event loop."""
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


async def _queue_drain_loop() -> None:
    """Фоновая корутина: ждёт тикеты блокирующе и стартует celery-задачи."""
    assert container is not None
    queue = await container.get(ILLMQueue)
    logger = await container.get(AISearchLogger)
    async_redis_storage = await container.get(KeyValueStorageProtocol)
    while True:
        try:
            item = await queue.dequeue_blocking(timeout=5)
            if not item:
                await asyncio.sleep(0.1)
                continue

            ticket_id, payload = item
            if (
                not isinstance(payload, dict)
                or "pack_key" not in payload
                or "result_key" not in payload
            ):
                await queue.set_failed(ticket_id, "Invalid payload: missing pack_key or result_key")
                await queue.ack(ticket_id)

                continue
            pack_key = payload["pack_key"]
            raw = await async_redis_storage.get(pack_key)
            pack = json.loads(raw)
            task_type = pack.get("type")

            if task_type == "search":
                worker.send_task(
                    "search_task",
                    args=(ticket_id, payload["pack_key"], payload["result_key"]),
                    queue="default",
                    task_id=ticket_id,
                )
            elif task_type == "generate":
                worker.send_task(
                    "generate_task",
                    args=(ticket_id, payload["pack_key"], payload["result_key"]),
                    queue="default",
                    task_id=ticket_id,
                )
        except Exception as e:
            logger.error(f"{e} - ошибка в drain queue loop")
            await asyncio.sleep(1)


@worker_process_init.connect
def on_worker_process_init(**kwargs: dict[str, tp.Any]) -> None:
    """Процесс перед инициализацией воркера"""
    global drain_task  # noqa: PLW0603

    asyncio.set_event_loop(loop)
    init_container_and_model()
    # Запуск drain loop в отдельной задаче
    drain_task = asyncio.ensure_future(_queue_drain_loop(), loop=loop)

    # Запуск loop в отдельном потоке
    def _start_loop() -> None:
        loop.run_forever()

    threading.Thread(target=_start_loop, daemon=True).start()


@task_prerun.connect
def on_task_prerun(task: tp.Callable, task_id: str, **kwargs: dict[str, tp.Any]) -> None:
    """Процесс перед выполнением задачи"""
    task._container = container
    task._model = model
    task.task_id = task_id


@worker_process_shutdown.connect
def on_worker_process_shutdown(**kwargs: dict[str, tp.Any]) -> None:
    """Процесс после остановки воркера"""

    async def _stop_drain() -> None:
        if drain_task:
            drain_task.cancel()
            with contextlib.suppress(Exception):
                await drain_task

    async def _run() -> None:
        await _stop_drain()
        if container:
            vllm_client = await container.get(IVLLMAdapter)
            await vllm_client.close()
            await container.close()

    run_coroutine(_run())
