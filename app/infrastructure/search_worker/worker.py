from __future__ import annotations
import asyncio
import threading
import traceback
import typing as tp
from asyncio import Future
from typing import TYPE_CHECKING

from celery import Celery
from celery.signals import (
    task_prerun,
    worker_process_init,
    worker_process_shutdown,
)

from app.services.interfaces import IHybridSearchOrchestrator
from pre_launch import load_collection_and_index

if TYPE_CHECKING:
    from dishka import AsyncContainer

from dishka import AsyncContainer, make_async_container

from app.common.logger import AISearchLogger, LoggerType

from app.infrastructure.adapters.interfaces import  IVLLMAdapter
from app.infrastructure.utils.nlp import init_nltk_resources
from app.settings.config import HybridSearchSettings, Settings, settings

container: AsyncContainer | None = None
logger: AISearchLogger | None = None
successful_warmup: bool = False

worker = Celery(
    "aisearch",
    broker=str(settings.redis.dsn),
    backend=str(settings.redis.dsn),
    include=["app.infrastructure.search_worker.tasks"],
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

worker.autodiscover_tasks(["app.infrastructure.search_worker.tasks"])

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
    global container  # noqa: PLW0603
    from app.infrastructure.ioc.search_ioc import ApplicationProvider
    from app.infrastructure.storages.milvus_provider import MilvusProvider
    from app.infrastructure.providers import (
        LoggerProvider,
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

    return container


def run_coroutine(coro: tp.Coroutine) -> Future:
    """Запуск корутин в общем event loop."""
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


@worker_process_init.connect
def on_worker_process_init(**kwargs: dict[str, tp.Any]) -> None:
    init_container_and_model()
    global successful_warmup

    try:
        from app.infrastructure.utils.nlp import init_nltk_resources
        init_nltk_resources()
        logger and logger.info("📚 NLTK ресурсы готовы")
    except Exception as e:
        logger and logger.warning(f"⚠️ Ошибка при инициализации NLTK: {e!r}")
    # стартуем event loop в отдельном потоке
    def _start_loop() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    threading.Thread(target=_start_loop, daemon=True).start()

    async def _eager_init() -> bool:
        orch = await container.get(IHybridSearchOrchestrator)
        try:
            ok = await orch.warmup()
            return ok
        except Exception as e:
            logger and logger.warning(f"⚠️ Warmup error: {e!r}")
            return False
    try:
     successful_warmup = run_coroutine(_eager_init())
     logger and logger.info(f"✅ Завершили прогрев, result={successful_warmup}")
    except Exception as e:
        logger and logger.warning(f"⚠️ Warmup failed with exception: {e!r}")
        successful_warmup = False


@task_prerun.connect
def on_task_prerun(task: tp.Callable, task_id: str, **kwargs: dict[str, tp.Any]) -> None:
    """Процесс перед выполнением задачи"""
    task._container = container
    task.task_id = task_id


@worker_process_shutdown.connect
def on_worker_process_shutdown(**kwargs: dict[str, tp.Any]) -> None:
    """Корректный shutdown воркера Celery."""


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
        await _shutdown_container_and_clients()
        # Останавливаем глобальный loop в отдельном потоке
        loop.call_soon_threadsafe(loop.stop)

    # Запускаем shutdown в глобальном loop и ждем завершения
    run_coroutine(_run_shutdown())



@worker.task(name='main_worker_health_check')
def health_check():
    """Healthcheck задача, которая проверяет готовность воркера"""
    global container
    
    checks = {
        "container_ready": container is not None,
        "event_loop_running": loop.is_running() if loop else False,
        "successful_warmup": successful_warmup,
    }
    
    all_healthy = all(checks.values())

    return {
        "status": "healthy" if all_healthy else "unhealthy",
        "checks": checks
    }