from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from dishka import make_async_container
from dishka.integrations.fastapi import FastapiProvider, setup_dishka
from fastapi import FastAPI

from app.api.v1.routers.base import router as base_router
from app.api.v1.routers.semantic_search import router as semantic_search_router
from app.api.v1.routers.taskmanager import router as taskmanager_router
from app.common.logger import LoggerType
from app.domain.exception_handler import exception_config
from app.infrastructure.ioc import ApplicationProvider
from app.infrastructure.providers import (
    AuthProvider,
    LoggerProvider,
    MilvusProvider,
    RedisProvider,
)
from app.infrastructure.utils.nlp import download_nltk_resources
from app.services.interfaces import ISemanticSearchService
from app.settings.config import (
    AppSettings,
    CelerySettings,
    MilvusSettings,
    RedisSettings,
    RestrictionSettings,
    Settings,
    settings,
)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Управление жизненным циклом приложения"""
    if not hasattr(application.state, "dishka_container"):
        raise RuntimeError("Не инициализирован контейнер зависимостей")
    container = application.state.dishka_container
    async with container() as request_container:
        await request_container.get(ISemanticSearchService)
        yield


def create_app() -> FastAPI:
    """Инициализация приложения"""
    application = FastAPI(title="AI Search", root_path=settings.app.prefix, lifespan=lifespan)
    container = make_async_container(
        ApplicationProvider(),
        FastapiProvider(),
        AuthProvider(),
        LoggerProvider(),
        MilvusProvider(),
        RedisProvider(),
        context={
            AppSettings: settings.app,
            MilvusSettings: settings.milvus,
            RedisSettings: settings.redis,
            LoggerType: LoggerType.APP,
            RestrictionSettings: settings.restrictions,
            CelerySettings: settings.celery,
            Settings: settings,
        },
    )
    setup_dishka(container, application)
    application.include_router(base_router)
    application.include_router(semantic_search_router)
    application.include_router(taskmanager_router)
    for exception, handler in exception_config.items():
        application.add_exception_handler(exception, handler)
    download_nltk_resources()
    return application


app = create_app()
