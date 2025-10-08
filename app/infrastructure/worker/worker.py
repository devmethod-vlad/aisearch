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
from sentence_transformers import CrossEncoder

from app.services.interfaces import IHybridSearchOrchestrator

if TYPE_CHECKING:
    from dishka import AsyncContainer

from dishka import AsyncContainer, make_async_container

from app.common.logger import AISearchLogger, LoggerType
from app.common.storages.interfaces import KeyValueStorageProtocol
from app.infrastructure.adapters.interfaces import ILLMQueue, IVLLMAdapter
from app.infrastructure.utils.nlp import download_nltk_resources, init_nltk_resources
from app.settings.config import HybridSearchSettings, Settings, settings

container: AsyncContainer | None = None
model: tp.Any | None = None
ce_model: tp.Any | None = None
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
    task_queues={"gpu-search": {"exchange": "gpu-search", "routing_key": "gpu-search"}},
    task_routes={
        "main_worker_health_check": {"queue": "gpu-search"},
        "search_task": {"queue": "gpu-search"},
        "generate-answer-vllm": {"queue": "gpu-search"},
    },
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
    init_nltk_resources()


def init_container_and_model() -> AsyncContainer:
    """Инициализация контейнера Dishka"""
    global container, model, ce_model  # noqa: PLW0603
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
    model.encode(["warmup"], convert_to_numpy=True)

    logger.info("Модель успешно загружена")
    ce_model = CrossEncoder(settings.reranker.model_name, device=settings.reranker.device)
    logger.info("Кросс-энкодер загружен")
    return container


def run_coroutine(coro: tp.Coroutine) -> Future:
    """Запуск корутин в общем event loop."""
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


@worker_process_init.connect
def on_worker_process_init(**kwargs: dict[str, tp.Any]) -> None:
    init_container_and_model()
    global drain_task, sweep_task

    # стартуем event loop в отдельном потоке
    def _start_loop() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    threading.Thread(target=_start_loop, daemon=True).start()


@task_prerun.connect
def on_task_prerun(task: tp.Callable, task_id: str, **kwargs: dict[str, tp.Any]) -> None:
    """Процесс перед выполнением задачи"""
    task._container = container
    task._model = model
    task._ce_model = ce_model
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



@worker.task(name='main_worker_health_check')
def health_check():
    """Healthcheck задача, которая проверяет готовность воркера"""
    global container, model, ce_model
    
    checks = {
        "container_ready": container is not None,
        "model_loaded": model is not None,
        "ce_model_loaded": ce_model is not None,
        "event_loop_running": loop.is_running() if loop else False
    }
    
    all_healthy = all(checks.values())
    
    return {
        "status": "healthy" if all_healthy else "unhealthy",
        "checks": checks
    }