import asyncio
from collections.abc import AsyncGenerator

import pytest
from faker import Faker
from httpx import ASGITransport, AsyncClient
from redis.asyncio import from_url
from redis.asyncio.client import Pipeline, Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer
from yoyo import get_backend, read_migrations

from app.common.database import Database
from app.common.exceptions.exceptions import RedisConnectionError
from app.common.redis import RedisHelper
from app.dependencies.dependencies import (
    get_db,
    get_redis,
    get_uow,
)
from app.infrastructure.unit_of_work.interfaces import IUnitOfWork
from app.infrastructure.unit_of_work.uow import UnitOfWork
from app.main import app
from app.services.glossary import GlossaryService
from app.services.interfaces import IGlossaryService
from app.settings.config import RedisSettings, Settings

pytest_plugins = ("tests.fixtures.glossary_element",)


@pytest.fixture(scope="session")
def event_loop():
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
def settings() -> Settings:
    """Фикстура настроек"""
    return Settings()


@pytest.fixture
def valid_token(settings: Settings) -> str:
    """Фикстура для корректного токена."""
    return settings.app.access_key


class TestDatabase:
    """Вспомогательный класс для работы с тестовой БД"""

    def __init__(self, dsn: str):
        self.engine = create_async_engine(url=str(dsn), echo=False, pool_size=5, max_overflow=10)

        self.session_factory = async_sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )


@pytest.fixture(scope="session")
async def db() -> AsyncGenerator[Database]:
    """Фикстура для инициализации тестовой базы данных."""
    container = PostgresContainer("postgres:13.4-alpine")
    container.start()

    test_db_url = (
        f"postgresql+asyncpg://{container.username}:{container.password}@"
        f"{container.get_container_host_ip()}:{container.get_exposed_port(5432)}/{container.dbname}"
    )
    database = TestDatabase(dsn=test_db_url)

    backend = get_backend(test_db_url.replace("postgresql+asyncpg", "postgresql"))
    migrations = read_migrations("app/infrastructure/migrations")
    assert migrations

    with backend.lock():
        backend.apply_migrations(backend.to_apply(migrations))

    yield database
    container.stop()


@pytest.fixture()
async def db_session(db: Database) -> AsyncGenerator[AsyncSession]:
    """Фикстура для создания изолированной сессии для каждого теста."""
    async with db.session_factory() as session, session.begin():
        yield session
        await session.rollback()


class TestRedisHelper:
    """Вспомогательный класс для работы с тестовым Redis"""

    def __init__(self, config: RedisSettings, url: str):
        self.config: RedisSettings = config
        self.connection: Redis = from_url(
            url,
            socket_connect_timeout=config.connect_timeout,
            socket_timeout=config.timeout,
            decode_responses=True,
        )

    async def pipe(self) -> Pipeline:
        """Получение redis pipeline с предварительной проверкой на доступ к redis"""
        try:
            await self.connection.ping()
        except (TimeoutError, ConnectionError):
            raise RedisConnectionError
        else:
            return self.connection.pipeline(transaction=True)


@pytest.fixture()
async def redis(settings: Settings) -> AsyncGenerator[RedisHelper]:
    """Фикстура для создания тестового подключения к Redis."""
    container = RedisContainer("redis:7.4.2-alpine")
    container.start()
    host = container.get_container_host_ip()
    port = container.get_exposed_port(6379)
    yield TestRedisHelper(config=settings.redis, url=f"redis://{host}:{port}")
    container.stop()


@pytest.fixture()
async def override_app_dependencies(db: Database, redis: RedisHelper) -> AsyncGenerator[None, None]:
    """Переопределение зависимостей приложения на тестовые ресурсы."""

    async def override_get_db() -> AsyncGenerator[Database, None]:
        yield db

    async def override_get_uow() -> AsyncGenerator[IUnitOfWork, None]:
        yield UnitOfWork(db)

    async def override_get_redis() -> AsyncGenerator[RedisHelper, None]:
        yield redis

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_uow] = override_get_uow
    app.dependency_overrides[get_redis] = override_get_redis

    yield

    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
async def cleanup_glossary_elements(db: Database) -> AsyncGenerator[None, None]:
    """Очистка таблицы glossary_element для изоляции тестовых данных."""

    async def clear() -> None:
        async with db.session_factory() as session:
            await session.execute(text("TRUNCATE TABLE glossary_element RESTART IDENTITY CASCADE"))
            await session.commit()

    await clear()
    yield
    await clear()


@pytest.fixture()
async def glossary_service(
    db: Database,
    redis: RedisHelper,
    override_app_dependencies,
) -> AsyncGenerator[IGlossaryService, None]:
    """Фикстура для тестового сервиса глоссария с переопределенной базой данных."""
    service = GlossaryService(UnitOfWork(db), redis)

    return service


@pytest.fixture()
async def client(
    override_app_dependencies,
) -> AsyncGenerator[AsyncClient]:
    """Фикстура для тестового клиента FastAPI с тестовыми зависимостями."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as test_client:
        yield test_client
