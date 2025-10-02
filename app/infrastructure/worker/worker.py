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
sweep_task: asyncio.Task | None = None


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
                try:
                    worker.send_task(
                        "search_task",
                        args=(ticket_id, payload["pack_key"], payload["result_key"]),
                        queue="default",
                        task_id=ticket_id,
                    )
                except Exception as e:
                    await queue.set_failed(ticket_id, f"send_task(search) error: {e!r}")
                    await queue.ack(ticket_id)

            elif task_type == "generate":
                try:
                    worker.send_task(
                        "generate_task",
                        args=(ticket_id, payload["pack_key"], payload["result_key"]),
                        queue="default",
                        task_id=ticket_id,
                    )
                except Exception as e:
                    await queue.set_failed(ticket_id, f"send_task(generate) error: {e!r}")
                    await queue.ack(ticket_id)

            else:

                await queue.set_failed(ticket_id, f"unknown task type: {task_type!r}")
                await queue.ack(ticket_id)
        except Exception as e:
            logger.error(f"{e} - ошибка в drain queue loop")
            await asyncio.sleep(1)


async def _processing_sweeper(period_sec: int = 10, stale_sec: int = 60) -> None:
    """Очистка протухших тикетов и перестановка в начало очереди"""
    assert container is not None
    queue = await container.get(ILLMQueue)
    _logger = await container.get(AISearchLogger)
    while True:
        try:
            n = await queue.sweep_processing(stale_sec=stale_sec)
            if n:
                _logger.warning(f"sweep_processing: requeued {n} stale tickets")
        except Exception as e:
            _logger.error(f"sweep_processing error: {e!r}")
        await asyncio.sleep(period_sec)


def spawn_supervised(
    name: str,
    coro_factory: tp.Callable[[], tp.Awaitable[None]],
    log,
    *,
    backoff_initial: float = 0.5,
    backoff_max: float = 30.0,
) -> Future:
    """
    Запускает корутину с автоперезапуском при крэше.
    coro_factory ДОЛЖНА возвращать НОВУЮ корутину при каждом вызове.
    """

    async def _runner() -> None:
        delay = backoff_initial
        while True:
            try:
                log.info(f"[supervisor] start {name}")
                await coro_factory()
                log.warning(f"[supervisor] {name} exited; restart in 1s")
                delay = backoff_initial
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                log.info(f"[supervisor] {name} cancelled")
                raise
            except Exception as e:
                log.exception(f"[supervisor] {name} crashed: {e!r}; restart in {delay:.1f}s")
                await asyncio.sleep(delay)
                delay = min(delay * 2, backoff_max)

    # важно: запускаем в глобальном loop через thread-safe API
    return asyncio.run_coroutine_threadsafe(_runner(), loop)


@worker_process_init.connect
def on_worker_process_init(**kwargs: dict[str, tp.Any]) -> None:
    init_container_and_model()
    global drain_task, sweep_task

    # стартуем event loop в отдельном потоке
    def _start_loop() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    threading.Thread(target=_start_loop, daemon=True).start()

    # теперь можно запускать фоновые supervised задачи
    drain_task = spawn_supervised("drain", lambda: _queue_drain_loop(), logger)
    sweep_task = spawn_supervised("sweeper", lambda: _processing_sweeper(), logger)


@task_prerun.connect
def on_task_prerun(task: tp.Callable, task_id: str, **kwargs: dict[str, tp.Any]) -> None:
    """Процесс перед выполнением задачи"""
    task._container = container
    task._model = model
    task.task_id = task_id


@worker_process_shutdown.connect
def on_worker_process_shutdown(**kwargs: dict[str, tp.Any]) -> None:
    """Корректный shutdown воркера Celery."""

    async def _stop_background_tasks() -> None:
        """Отмена и завершение drain_task и sweep_task."""
        tasks = [t for t in (drain_task, sweep_task) if t]
        for t in tasks:
            t.cancel()

        for t in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t

    async def _shutdown_container_and_clients() -> None:
        if container:
            # Закрываем VLLM-клиент
            try:
                vllm_client = await container.get(IVLLMAdapter)
                await vllm_client.close()
            except Exception as e:
                logger and logger.warning(f"Ошибка при закрытии VLLM-клиента: {e!r}")

            # Закрываем сам контейнер
            try:
                await container.close()
            except Exception as e:
                logger and logger.warning(f"Ошибка при закрытии контейнера: {e!r}")

    async def _run_shutdown() -> None:
        await _stop_background_tasks()
        await _shutdown_container_and_clients()
        # Останавливаем глобальный loop в отдельном потоке
        loop.call_soon_threadsafe(loop.stop)

    # Запускаем shutdown в глобальном loop и ждем завершения
    run_coroutine(_run_shutdown())
