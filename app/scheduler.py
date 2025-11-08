import asyncio
from dishka import make_async_container

from apscheduler.schedulers.asyncio import AsyncIOScheduler


from app.common.logger import LoggerType, AISearchLogger
from app.infrastructure.ioc.search_ioc import ApplicationProvider
from app.infrastructure.storages.milvus_provider import MilvusProvider
from app.infrastructure.providers import (
    LoggerProvider,
    RedisProvider,
)
from app.services.interfaces import IUpdaterService
from app.settings.config import (
    AppSettings,
    LLMGlobalSemaphoreSettings,
    LLMQueueSettings,
    MilvusSettings,
    RedisSettings,
    VLLMSettings,
    settings,
    Settings,
    HybridSearchSettings,
)


async def update_vio(service: IUpdaterService):
    """Обновление ВИО бз"""
    print("ВОПРОСЫ И ОТВЕТЫ")
    await service.update_vio_base()


async def update_kb(service: IUpdaterService):
    """Обновление БЗ ТП"""
    print("КБ_ВИКИ")
    await service.update_kb_base()


async def main():
    """Точка входа для шедулера"""
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
    scheduler = AsyncIOScheduler(
        job_defaults={
            "misfire_grace_time": 300,  # 5 минут на запуск пропущенной задачи
            "coalesce": True,  # Объединять пропущенные выполнения
            "max_instances": 1,  # Только один экземпляр задачи одновременно
        }
    )
    # scheduler.add_job(
    #     update_vio,
    #     "cron",
    #     hour=settings.extract_edu.cron_update_hours,
    #     args=(service,),
    #     id="update_vio",
    #     replace_existing=True,
    # )
    # scheduler.add_job(
    #     update_kb,
    #     "cron",
    #     hour=settings.extract_edu.cron_update_hours,
    #     args=(service,),
    #     id="update_kb",
    #     replace_existing=True,
    # )
    scheduler.add_job(
        update_vio,
        "interval",
        seconds=60,
        args=(service,),
        id="update_vio",
        replace_existing=True,
    )
    # scheduler.add_job(
    #     update_kb,
    #     "interval",
    #     seconds=60,
    #     args=(service,),
    #     id="update_kb",
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
