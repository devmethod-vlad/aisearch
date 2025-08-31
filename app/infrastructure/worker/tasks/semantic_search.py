import asyncio
import typing as tp

from dishka import AsyncContainer
from sentence_transformers import SentenceTransformer

from app.common.logger import AISearchLogger
from app.infrastructure.storages.interfaces import IVectorDatabase
from app.infrastructure.utils.nlp import get_documents
from app.infrastructure.worker.celery_worker import clear_cache, worker
from app.settings.config import RestrictionSettings


@worker.task(name="semantic-search")
def task_semantic_search(
    collection_name: str, query: str, query_key: str, top_k: int
) -> list[dict[str, tp.Any]]:
    """Задача для семантического поиска"""
    container: AsyncContainer | None = getattr(task_semantic_search, "container", None)
    if not container:
        raise Exception("Не удалось инициализировать контейнер Dishka")

    model: SentenceTransformer | None = getattr(task_semantic_search, "model", None)
    if not model:
        raise Exception("Не удалось инициализировать модель")

    async def run_task() -> None:
        async with container() as request_container:

            logger = await request_container.get(AISearchLogger)
            logger.info("Начало выполнения задачи 'semantic-search'...")

            restrictions_settings = await request_container.get(RestrictionSettings)

            vector_db = await request_container.get(IVectorDatabase)
            model_was_updated = await vector_db.ensure_model_consistency(
                collection_name=collection_name, model=model, documents=get_documents()
            )
            if model_was_updated:
                await asyncio.to_thread(
                    clear_cache, f"{restrictions_settings.base_cache_key}:{query_key}"
                )

            query_vector = model.encode(query)
            result = await vector_db.search(collection_name, query_vector.tolist(), top_k)
            logger.info("Задача 'semantic-search' выполнена")

            return result

    return asyncio.run(run_task())
