import typing as tp
from contextlib import asynccontextmanager

from dishka import make_async_container
from dishka.integrations.fastapi import FastapiProvider, setup_dishka
from fastapi import FastAPI

from app.api.v1.routers.base import router as base_router
from app.api.v1.routers.hybrid_search import router as hybrid_router
from app.api.v1.routers.taskmanager import router as taskmanager_router
from app.common.logger import AISearchLogger, LoggerType
from app.domain.exception_handler import exception_config
from app.infrastructure.ioc import ApplicationProvider
from app.infrastructure.providers import (
    AuthProvider,
    LoggerProvider,
    RedisProvider,
)
from app.infrastructure.utils.nlp import download_nltk_resources
from app.settings.config import (
    AppSettings,
    CelerySettings,
    HybridSearchSettings,
    RedisSettings,
    Settings,
    settings,
)
from pre_launch import load_collection_and_index


@asynccontextmanager
async def lifespan(application: FastAPI) -> tp.AsyncGenerator[None, None]:
    """Управление жизненным циклом приложения"""
    if not hasattr(application.state, "dishka_container"):
        raise RuntimeError("Не инициализирован контейнер зависимостей")
    container = application.state.dishka_container
    async with container() as request_container:
        logger = await request_container.get(AISearchLogger)
        download_nltk_resources()
        await load_collection_and_index(settings=settings, logger=logger)
        yield


def create_app() -> FastAPI:
    """Инициализация приложения"""
    application = FastAPI(title="AI Search", root_path=settings.app.prefix, lifespan=lifespan)
    container = make_async_container(
        ApplicationProvider(),
        FastapiProvider(),
        AuthProvider(),
        LoggerProvider(),
        RedisProvider(),
        context={
            AppSettings: settings.app,
            RedisSettings: settings.redis,
            LoggerType: LoggerType.APP,
            CelerySettings: settings.celery,
            HybridSearchSettings: settings.hybrid,
            Settings: settings,
        },
    )
    setup_dishka(container, application)
    application.include_router(base_router)
    application.include_router(taskmanager_router)
    application.include_router(hybrid_router)
    for exception, handler in exception_config.items():
        application.add_exception_handler(exception, handler)

    return application


app = create_app()
