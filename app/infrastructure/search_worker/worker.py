from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import typing as tp

import redis
from celery import Celery
from celery.signals import (
    setup_logging as celery_setup_logging,
    task_prerun,
    worker_process_init,
    worker_process_shutdown,
)
from dishka import AsyncContainer, make_async_container

from app.common.logger import LoggerType
from app.common.storages.sync_redis import SyncRedisStorage
from app.infrastructure.utils.nlp import init_nltk_resources
from app.infrastructure.utils.process import (
    update_process_info,
)
from app.services.interfaces import IHybridSearchOrchestrator
from app.settings.config import (
    AppSettings,
    HybridSearchSettings,
    LLMGlobalSemaphoreSettings,
    LLMQueueSettings,
    MilvusSettings,
    PostgresSettings,
    RedisSettings,
    Settings,
    settings,
)
from app.settings.logging_config import setup_logging

w_id = os.getenv("WORKER_ID")
container: AsyncContainer | None = None
successful_warmup: bool = False
sync_redis_storage = SyncRedisStorage(client=redis.from_url(settings.redis.dsn))


def get_container_from_task(task_instance: tp.Callable) -> AsyncContainer:
    """Получает оригинальный AsyncContainer из экземпляра задачи Celery.
    Вызывает исключение, если контейнер не найден.
    """
    original_container: AsyncContainer | None = getattr(
        task_instance, "_container", None
    )
    if not original_container:
        raise Exception("Не удалось инициализировать контейнер Dishka")
    return original_container


@celery_setup_logging.connect
def config_loggers(*args: tuple[tp.Any], **kwargs: dict[str, tp.Any]) -> None:
    """Настройка логирования для Celery через сигнал."""
    setup_logging()
    logger = logging.getLogger("celery")
    logger.info("Логирование Celery инициализировано через сигнал.")


worker = Celery(
    "aisearch",
    broker=str(settings.redis.dsn),
    backend=str(settings.redis.dsn),
    include=["app.infrastructure.search_worker.tasks"],
    worker_proc_alive_timeout=120,
)

worker.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    task_queues={"gpu-search": {"exchange": "gpu-search", "routing_key": "gpu-search"}},
    task_routes={
        "search_task": {"queue": "gpu-search"},
    },
    result_expires=600,
)

worker.autodiscover_tasks(["app.infrastructure.search_worker.tasks"])


def init_container_and_model() -> AsyncContainer:
    """Инициализация контейнера Dishka"""
    global container  # noqa: PLW0603
    from app.infrastructure.ioc.search_ioc import ApplicationProvider
    from app.infrastructure.providers import (
        LoggerProvider,
        RedisProvider,
    )
    from app.infrastructure.storages.milvus_provider import MilvusProvider

    container = make_async_container(
        ApplicationProvider(),
        LoggerProvider(),
        MilvusProvider(),
        RedisProvider(),
        context={
            AppSettings: settings.app,
            Settings: settings,
            MilvusSettings: settings.milvus,
            PostgresSettings: settings.postgres,
            RedisSettings: settings.redis,
            LoggerType: LoggerType.CELERY,
            HybridSearchSettings: settings.hybrid,
            LLMGlobalSemaphoreSettings: settings.llm_global_sem,
            LLMQueueSettings: settings.llm_queue,
        },
    )

    return container


global_loop = asyncio.new_event_loop()


def run_in_loop(coro: tp.Awaitable[tp.Any]) -> tp.Any:
    """Запуск корутин в общем event loop."""
    return global_loop.run_until_complete(coro)


def run_coroutine(coro: tp.Awaitable[tp.Any]) -> tp.Any:
    """Запуск корутин в общем event loop в потоке."""
    return asyncio.run_coroutine_threadsafe(coro, global_loop).result()


@worker_process_init.connect
def on_worker_process_init(**kwargs: dict[str, tp.Any]) -> None:
    global successful_warmup  # noqa: PLW0603

    update_process_info(
        key=f"{w_id}:celery-proc",
        info=health_check(),
        sync_redis_storage=sync_redis_storage,
    )

    init_container_and_model()

    try:
        init_nltk_resources()
        logger = logging.getLogger("celery")
        logger.info("📚 NLTK ресурсы готовы")
    except Exception as e:
        logger = logging.getLogger("celery")
        logger.warning(f"⚠️ Ошибка при инициализации NLTK: {e!r}")

    def _start_loop() -> None:
        asyncio.set_event_loop(global_loop)
        global_loop.run_forever()

    threading.Thread(target=_start_loop, daemon=True).start()

    async def _eager_init() -> bool:
        orch = await container.get(IHybridSearchOrchestrator)
        try:
            ok = await orch.warmup()
            return ok
        except Exception as e:
            logger = logging.getLogger("celery")
            logger.warning(f"⚠️ Warmup error: {e!r}")
            return False

    try:
        successful_warmup = run_coroutine(_eager_init())
        logger = logging.getLogger("celery")
        logger.info(f"✅ Завершили прогрев, result={successful_warmup}")
    except Exception as e:
        logger = logging.getLogger("celery")
        logger.warning(f"⚠️ Warmup failed with exception: {e!r}")
        successful_warmup = False

    update_process_info(
        key=f"{w_id}:celery-proc",
        info=health_check(),
        sync_redis_storage=sync_redis_storage,
    )


@task_prerun.connect
def on_task_prerun(
    task: tp.Callable, task_id: str, **kwargs: dict[str, tp.Any]
) -> None:
    """Процесс перед выполнением задачи"""
    task._container = container
    task.task_id = task_id


@worker_process_shutdown.connect
def on_worker_process_shutdown(**kwargs: dict[str, tp.Any]) -> None:
    """Корректный shutdown воркера Celery."""

    async def _shutdown_container_and_clients() -> None:
        if container:
            try:
                await container.close()
            except Exception as e:
                logger = logging.getLogger("celery")
                logger.warning(f"Ошибка при закрытии контейнера: {e!r}")

    async def _run_shutdown() -> None:
        await _shutdown_container_and_clients()

        global_loop.call_soon_threadsafe(global_loop.stop)

    run_coroutine(_run_shutdown())


def health_check() -> dict:
    """Healthcheck, который проверяет готовность воркера"""
    checks = {
        "container_ready": container is not None,
        "event_loop_running": global_loop.is_running() if global_loop else False,
        "successful_warmup": successful_warmup,
    }

    all_healthy = all(checks.values())

    health_info = {
        "status": "healthy" if all_healthy else "unhealthy",
        "checks": checks,
    }
    return {
        "all_healthy": int(health_info["status"] == "healthy"),
        "checks": json.dumps(health_info["checks"]),
    }
