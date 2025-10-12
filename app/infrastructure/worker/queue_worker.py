from __future__ import annotations
import asyncio
import json
import typing as tp

from celery import Celery

from dishka import AsyncContainer, make_async_container

from app.common.storages.interfaces import KeyValueStorageProtocol
from app.infrastructure.adapters.interfaces import ILLMQueue
from app.common.logger import AISearchLogger, LoggerType
from app.settings.config import settings, Settings, HybridSearchSettings

container: AsyncContainer | None = None
logger: AISearchLogger | None = None

# ---- Celery client (лёгкий) ----
celery_client = Celery(
    "queue-pump",
    broker=str(settings.redis.dsn),      # redis://.../7  ← брокер Celery
    backend=str(settings.redis.dsn),     # можно вынести в db 8
)
celery_client.conf.update(
    task_create_missing_queues=True,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
)

def init_container() -> AsyncContainer:
    global container, logger
    from app.infrastructure.ioc import ApplicationProvider
    from app.infrastructure.providers import LoggerProvider, MilvusProvider, RedisProvider
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
    logger = AISearchLogger(logger_type=LoggerType.CELERY)
    return container

async def _queue_drain_loop() -> None:
    """Фоновая корутина: ждёт тикеты и шлёт их в Celery"""
    assert container is not None
    queue = await container.get(ILLMQueue)
    redis_storage = await container.get(KeyValueStorageProtocol)
    logger = await container.get(AISearchLogger)
    logger.info("🚩 Старт корутины работы с очередью")
    while True:
        item = await queue.dequeue_blocking(timeout=1)
        if not item:
            await asyncio.sleep(0.1)
            continue

        ticket_id, payload = item
        if not isinstance(payload, dict) or "pack_key" not in payload or "result_key" not in payload:
            logger.error(f"⚠️ Проблема с полезной нагрузкой тикета {ticket_id}, payload: {payload}")
            await queue.set_failed(ticket_id, "⚠️ Invalid payload")
            await queue.ack(ticket_id)
            continue

        pack_key = payload["pack_key"]
        raw = await redis_storage.get(pack_key)
        if raw is None:
            await queue.set_failed(ticket_id, "⚠️ Missing pack")
            await queue.ack(ticket_id)
            continue

        try:
            task_type = json.loads(raw).get("type")
            if task_type == "search":
                logger.info("🚩 НАЧАЛО ВЫПОЛНЕНИЯ ЗАДАЧИ типа 'search'")
                celery_client.send_task(
                    "search_task",
                    args=(ticket_id, payload["pack_key"], payload["result_key"]),
                    queue="gpu-search",
                    task_id=ticket_id  # удобно связывать тикет и task_id
                )

            elif task_type == "generate":
                logger.info("🚩 НАЧАЛО ВЫПОЛНЕНИЯ ЗАДАЧИ типа 'generate'")
                celery_client.send_task(
                    "generate-answer-vllm",
                    args=(ticket_id, payload["pack_key"], payload["result_key"]),
                    queue="gpu-search",
                    task_id=ticket_id
                )
            else:
                logger.info(f"⚠️ Неизвестный тип задачи: {task_type}")
                await queue.set_failed(ticket_id, f"unknown task type: {task_type}")
        except Exception as e:
            logger.error(f"⚠️ ОШИБКА: {e}")
            await queue.set_failed(ticket_id, f"⚠️ send_task error: {e}")
        finally:
            logger.info(f"🧹 УБИРАЕМ ТИКЕТ ИЗ PROCESSING {ticket_id}")
            await queue.ack(ticket_id)

async def _processing_sweeper(period_sec: int = 10, stale_sec: int = 60) -> None:
    """Переставляет протухшие тикеты обратно в очередь"""
    assert container is not None
    queue = await container.get(ILLMQueue)
    logger = await container.get(AISearchLogger)
    logger.info("🚩 Старт корутины очистки застрявших тикетов")
    while True:
        try:
            n = await queue.sweep_processing(stale_sec=stale_sec)
            if n:
                logger.warning(f"🚨 Корутина sweep_processing: переставила {n} подвисших тикетов")
        except Exception as e:
            logger.error(f"⚠️ sweep_processing error: {e}")
        await asyncio.sleep(period_sec)

async def main_async() -> None:
    init_container()
    drain_task = asyncio.create_task(_queue_drain_loop())
    sweeper_task = asyncio.create_task(_processing_sweeper())
    logger = await container.get(AISearchLogger)

    try:
        await asyncio.gather(drain_task, sweeper_task)
    except asyncio.CancelledError as e:
        logger.error(f"⚠️ Останавливаем запущенные корутины: {e}")
        drain_task.cancel()
        sweeper_task.cancel()
        await asyncio.gather(drain_task, sweeper_task, return_exceptions=True)

if __name__ == "__main__":
    asyncio.run(main_async())
