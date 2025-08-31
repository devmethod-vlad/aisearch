import asyncio
import typing as tp

import redis
from celery import Celery
from celery.signals import (
    task_postrun,
    task_prerun,
    worker_process_init,
    worker_process_shutdown,
)
from dishka import AsyncContainer, Scope, make_async_container
from sentence_transformers import SentenceTransformer

from app.common.logger import AISearchLogger, LoggerType
from app.common.storages.sync_redis import SyncRedisStorage
from app.infrastructure.utils.nlp import download_nltk_resources
from app.infrastructure.worker.scheduler import get_schedule_config
from app.settings.config import settings

worker = Celery(__name__)

worker.conf.broker_url = str(settings.redis.dsn)
worker.conf.result_backend = str(settings.redis.dsn)

worker.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Europe/Moscow",
)

container: AsyncContainer | None = None
sync_redis_storage = SyncRedisStorage(client=redis.from_url(settings.redis.dsn))  # type: ignore

model: SentenceTransformer | None = None

worker.autodiscover_tasks(["app.infrastructure.worker.tasks.semantic_search"])
worker.autodiscover_tasks(["app.infrastructure.worker.tasks.knowledge_base"])

worker.conf.beat_schedule = get_schedule_config()
worker.conf.beat_max_loop_interval = 30
worker.conf.beat_schedule_filename = "volumes/celerybeat-schedule"


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
        MilvusSettings,
        RedisSettings,
        RestrictionSettings,
    )

    container = make_async_container(
        ApplicationProvider(scope=Scope.APP),
        LoggerProvider(),
        MilvusProvider(),
        RedisProvider(),
        context={
            AppSettings: settings.app,
            MilvusSettings: settings.milvus,
            RedisSettings: settings.redis,
            LoggerType: LoggerType.CELERY,
            RestrictionSettings: settings.restrictions,
        },
    )
    logger = AISearchLogger(logger_type=LoggerType.CELERY)

    logger.info(f"Выполняется загрузка модели {settings.app.model_name} ...")
    model = SentenceTransformer(settings.app.model_name)
    logger.info("Модель успешно загружена")

    logger.info("Выполняется загрузка ресурсов nltk ...")
    download_nltk_resources()
    logger.info("Загрузка ресурсов nltk завершена")


@worker_process_init.connect
def on_worker_process_init(**kwargs: dict[str, tp.Any]) -> None:
    """Создание Dishka контейнера перед выполнением задачи"""
    init_container_and_model()


@task_prerun.connect
def on_task_prerun(task: tp.Callable, task_id: str, **kwargs: dict[str, tp.Any]) -> None:
    """Опрокидывание контейнера в задачу"""
    task.container = container
    task.model = model
    sync_redis_storage.append(settings.restrictions.queue_key, task_id)


@task_postrun.connect
def on_task_postrun(task_id: str, **kwargs: dict[str, tp.Any]) -> None:
    """Удаление задачи из списка выполняемых задач"""
    existing_keys: list[str] | None = sync_redis_storage.list_range(
        key=settings.restrictions.queue_key, start=0, end=-1
    )
    if existing_keys:
        sync_redis_storage.delete(settings.restrictions.queue_key)
        existing_keys.remove(task_id)
        if existing_keys:
            sync_redis_storage.append(settings.restrictions.queue_key, *existing_keys)


@worker_process_shutdown.connect
def on_worker_process_shutdown(**kwargs: dict[str, tp.Any]) -> None:
    """Очистка Dishka контейнера после выполнения задачи"""
    if container:
        asyncio.run(container.close())


def clear_cache(*additional_keys: str) -> None:
    """Очистка информации о задачах в Redis"""
    sync_redis_storage.delete(
        *[
            *additional_keys,
            *[
                key_name.decode()
                for key_name in sync_redis_storage.scan_iter(match="celery-task-meta-*")
            ],
            settings.restrictions.queue_key,
        ],
    )
