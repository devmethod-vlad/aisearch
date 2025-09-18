import asyncio
from collections.abc import AsyncGenerator, Generator

import pytest
from dishka import AsyncContainer, make_async_container
from dishka.integrations.fastapi import FastapiProvider, setup_dishka
from faker import Faker
from fastapi import FastAPI
from httpx import AsyncClient
from pydantic import RedisDsn
from testcontainers.redis import RedisContainer

from app.api.v1.routers.base import router as base_router
from app.api.v1.routers.semantic_search import router as semantic_search_router
from app.api.v1.routers.taskmanager import router as taskmanager_router
from app.common.logger import LoggerType
from app.infrastructure.ioc import ApplicationProvider
from app.infrastructure.providers import AuthProvider, LoggerProvider, RedisProvider
from app.services.interfaces import ISemanticSearchService
from app.settings.config import (
    AppSettings,
    RedisSettings,
    Settings,
)


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop]:
    """Overrides pytest default function scoped event loop"""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture()
def faker() -> Faker:
    """Фикстура для наполнения тестовых данных"""
    return Faker()


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Фикстура для настройки anyio."""
    return "asyncio"


@pytest.fixture(scope="session")
def redis_container() -> Generator[RedisContainer]:
    """Фикстура для контейнера Redis."""
    container = RedisContainer("redis:7.4.2-alpine")
    container.start()
    yield container
    container.stop()


@pytest.fixture(scope="session")
def settings(redis_container: RedisContainer) -> Settings:
    """Фикстура настроек с тестовыми контейнерами."""
    redis_port = redis_container.get_exposed_port(6379)
    redis_host = redis_container.get_container_host_ip()

    return Settings(
        app=AppSettings(
            mode="dev",
            host="0.0.0.0",
            port=8000,
            debug_host="0.0.0.0",
            debug_port=8001,
            workers_num=1,
            prefix="",
            confluence_url="https://edu.emias.ru",
            edu_emias_url="https://edu.emias.ru",
            knowledge_base_minitable_google_link="minitable",
            knowledge_base_megatable_google_link="megatable",
        ),
        redis=RedisSettings(port=redis_port, dsn=RedisDsn(f"redis://{redis_host}:{redis_port}")),
    )


@pytest.fixture(scope="session")
async def container(settings: Settings) -> AsyncContainer:
    """Фикстура контейнера зависимостей"""
    # Mock* - наследован от интерфейса, MockOverriden* - унаследован от основого класса

    container = make_async_container(
        ApplicationProvider(),
        FastapiProvider(),
        AuthProvider(),
        RedisProvider(),
        LoggerProvider(),
        context={
            AppSettings: settings.app,
            RedisSettings: settings.redis,
            LoggerType: LoggerType.TEST,
        },
    )
    return container


@pytest.fixture()
async def request_container(container: AsyncContainer) -> AsyncGenerator[AsyncContainer]:
    """Контейнер уровня запросов."""
    async with container() as request_container:
        yield request_container


@pytest.fixture
async def semantic_search_service(request_container: AsyncContainer) -> ISemanticSearchService:
    """Сервис семантического поиска"""
    return await request_container.get(ISemanticSearchService)


@pytest.fixture(scope="session")
async def app(container: AsyncContainer) -> FastAPI:
    """Фикстура для создания тестового приложения FastAPI."""

    def create_test_app() -> FastAPI:
        """Инициализация тестового приложения."""
        application = FastAPI(title="Auth Service")
        setup_dishka(container, application)
        application.include_router(base_router)
        application.include_router(semantic_search_router)
        application.include_router(taskmanager_router)

        return application

    return create_test_app()


@pytest.fixture()
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient]:
    """Фикстура для тестового клиента FastAPI."""
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client
