from __future__ import annotations
import asyncio
import threading
import typing as tp
from asyncio import Future
from typing import TYPE_CHECKING
import os, socket
from multiprocessing import current_process
from celery import Celery
from celery.signals import (
    task_prerun,
    worker_process_init,
    worker_process_shutdown,
)
import psutil
import redis
from celery.signals import (
    celeryd_init
)
from app.common.storages.sync_redis import SyncRedisStorage
from app.services.interfaces import IHybridSearchOrchestrator

if TYPE_CHECKING:
    from dishka import AsyncContainer

from dishka import AsyncContainer, make_async_container

from app.common.logger import AISearchLogger, LoggerType

from app.infrastructure.adapters.interfaces import IVLLMAdapter
from app.infrastructure.utils.nlp import init_nltk_resources
from app.settings.config import HybridSearchSettings, Settings, settings

w_id = os.getenv("WORKER_ID")
container: AsyncContainer | None = None
logger: AISearchLogger | None = None
successful_warmup: bool = False
sync_redis_storage = SyncRedisStorage(client=redis.from_url(settings.redis.dsn))

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
        "generate-answer-vllm": {"queue": "gpu-search"},
    },
    result_expires=600,
)

worker.autodiscover_tasks(["app.infrastructure.search_worker.tasks"])

# --- Единый event loop ---
loop = asyncio.new_event_loop()


@celeryd_init.connect
def on_init(**kwargs: dict[str, tp.Any]) -> None:
    """Код, выполняющийся перед созданием воркера"""
    remove_process_keys()


@worker_process_shutdown.connect
def on_worker_process_shutdown(**kwargs: dict[str, tp.Any]) -> None:
    """Код, выполняющийся в каждом дочернем процессе при его завершении"""
    sync_redis_storage.delete(get_process_info()[0])


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


def remove_process_keys() -> int:
    """Очистка ключей воркеров"""
    keys_to_delete = []
    for key in sync_redis_storage.scan_iter(
        match=f"aisearch:health:{w_id}:celery-proc:*"
    ):
        if isinstance(key, bytes):
            keys_to_delete.append(key.decode())
        else:
            keys_to_delete.append(str(key))

    if keys_to_delete:
        return sync_redis_storage.delete(*keys_to_delete)

    return 0


def get_process_info() -> tuple[str, str, float]:
    """Получение информации о процессе"""
    wid_container = socket.gethostname()
    wid_proc_name = current_process().name
    wid_pid = os.getpid()
    proc = psutil.Process(wid_pid)
    proc_create_time = proc.create_time()
    return (
        f"aisearch:health:{w_id}:celery-proc:{wid_container}:{wid_proc_name}",
        wid_pid,
        proc_create_time,
    )


def update_process_info() -> None:
    """Обновление информации о процессе"""
    pkey, pid, ptime = get_process_info()
    health = health_check()["status"] == "healthy"

    sync_redis_storage.client.hset(
        pkey,
        mapping={
            "all_healthy": int(health),
            "pid": pid,
            "proc_created_at": ptime,
        },
    )


@worker_process_init.connect
def on_worker_process_init(**kwargs: dict[str, tp.Any]) -> None:
    global successful_warmup

    update_process_info()
    init_container_and_model()

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

    update_process_info()


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


def health_check():
    """Healthcheck, который проверяет готовность воркера"""
    checks = {
        "container_ready": container is not None,
        "event_loop_running": loop.is_running() if loop else False,
        "successful_warmup": successful_warmup,
    }

    all_healthy = all(checks.values())

    return {"status": "healthy" if all_healthy else "unhealthy", "checks": checks}
