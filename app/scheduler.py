import asyncio
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dishka import make_async_container

from app.common.logger import AISearchLogger, LoggerType
from app.infrastructure.adapters.interfaces import IEduAdapter
from app.infrastructure.ioc.search_ioc import ApplicationProvider
from app.infrastructure.providers import (
    LoggerProvider,
    RedisProvider,
)
from app.infrastructure.storages.milvus_provider import MilvusProvider
from app.services.interfaces import IUpdaterService
from app.settings.config import (
    AppSettings,
    HybridSearchSettings,
    LLMGlobalSemaphoreSettings,
    LLMQueueSettings,
    MilvusSettings,
    PostgresSettings,
    RedisSettings,
    Settings,
    VLLMSettings,
    settings,
)
from app.settings.logging_config import setup_logging


async def update_vio(service: IUpdaterService, logger: logging.Logger) -> None:
    """Обновление ВИО БЗ"""
    logger.info("Обновление ВИО БЗ ...")
    await service.update_vio_base()
    logger.info("Обновление ВИО БЗ завершено")


async def update_kb(service: IUpdaterService, logger: logging.Logger) -> None:
    """Обновление БЗ ТП"""
    logger.info("Обновление БЗ ТП ...")
    await service.update_kb_base()
    logger.info("Обновление БЗ ТП завершено")


async def update_all_sources(
    service: IUpdaterService, edu_adapter: IEduAdapter, logger: logging.Logger
) -> None:
    """Обновление всех источников"""
    success = await edu_adapter.provoke_harvest_to_edu(harvest_type="all")
    if success:
        await service.update_all()
        logger.info("✅ Обновление источников завершено")
    else:
        logger.error("⚠️ Обновление источников не было завершено")


async def main() -> None:
    """Точка входа для шедулера"""
    setup_logging()
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
            LoggerType: LoggerType.UPDATER,
            HybridSearchSettings: settings.hybrid,
            LLMGlobalSemaphoreSettings: settings.llm_global_sem,
            LLMQueueSettings: settings.llm_queue,
            VLLMSettings: settings.vllm,
        },
    )
    logger = await container.get(AISearchLogger)
    logger.info("Запуск планировщика")
    service = await container.get(IUpdaterService)
    edu_adapter = await container.get(IEduAdapter)
    scheduler = AsyncIOScheduler(
        job_defaults={
            "misfire_grace_time": 300,  # 5 минут на запуск пропущенной задачи
            "coalesce": True,  # Объединять пропущенные выполнения
            "max_instances": 1,  # Только один экземпляр задачи одновременно
        }
    )

    for time_str in settings.extract_edu.cron_update_times.split(","):
        hour, minute = map(int, time_str.strip().split(":"))
        task_name = f"update_all_{hour:02d}_{minute:02d}"
        scheduler.add_job(
            update_all_sources,
            "cron",
            hour=hour,
            minute=minute,
            args=(service, edu_adapter, logger),
            name=task_name,
            id=task_name,
            replace_existing=True,
        )

    # scheduler.add_job(
    #     update_all,
    #     "interval",
    #     seconds=60,
    #     args=(service, edu_adapter, logger),
    #     id="update_vio",
    #     replace_existing=True,
    # )

    scheduler.start()
    logger.info("Планировщик успешно запущен")
    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Остановка планировщика")
        scheduler.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
