import asyncio
import typing as tp

from dishka import AsyncContainer

from app.common.logger import AISearchLogger
from app.infrastructure.worker.celery_worker import worker
from app.services.interfaces import IKnowledgeBaseService


@worker.task(name="collect-all-data-from-confluence")
def task_collect_all_data_from_confluence() -> list[dict[str, tp.Any]]:
    """Задача для семантического поиска"""
    container: AsyncContainer | None = getattr(
        task_collect_all_data_from_confluence, "container", None
    )
    if not container:
        raise Exception("Не удалось инициализировать контейнер Dishka")

    async def run_task() -> None:
        async with container() as request_container:

            logger = await request_container.get(AISearchLogger)
            logger.info("Начало выполнения задачи 'collect-data-from-confluence'...")

            knowledge_base_service = await request_container.get(IKnowledgeBaseService)

            result = (await knowledge_base_service.collect_all_data_from_confluence()).model_dump()
            logger.info("Задача 'semantic-search' выполнена")

            return result

    return asyncio.run(run_task())


@worker.task(name="collect-data-from-confluence-by-page-id")
def task_collect_data_from_confluence_by_page_id_task(page_id: int) -> list[dict[str, tp.Any]]:
    """Задача для семантического поиска"""
    container: AsyncContainer | None = getattr(
        task_collect_data_from_confluence_by_page_id_task, "container", None
    )
    if not container:
        raise Exception("Не удалось инициализировать контейнер Dishka")

    async def run_task() -> None:
        async with container() as request_container:

            logger = await request_container.get(AISearchLogger)
            logger.info("Начало выполнения задачи 'collect-data-from-confluence'...")

            knowledge_base_service = await request_container.get(IKnowledgeBaseService)

            result = (
                await knowledge_base_service.collect_data_from_confluence_by_page_id(
                    page_id=page_id
                )
            ).model_dump()

            logger.info("Задача 'semantic-search' выполнена")
            return result

    return asyncio.run(run_task())
