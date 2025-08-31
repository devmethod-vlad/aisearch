import redis.asyncio as redis
from dishka import Provider, Scope, from_context, provide
from fastapi import Request
from fastapi.security import HTTPAuthorizationCredentials

from app.common.auth import AccessBearer
from app.common.logger import AISearchLogger, LoggerType
from app.infrastructure.storages.milvus import MilvusDatabase
from app.settings.config import MilvusSettings, RedisSettings


class AuthProvider(Provider):
    """Провайдер для аутентификации."""

    @provide(scope=Scope.REQUEST)
    async def get_auth(self, request: Request) -> HTTPAuthorizationCredentials | None:
        """Получение аутентификации."""
        bearer = AccessBearer()
        return await bearer(request)


class MilvusProvider(Provider):
    """Провайдер для MilvusDB."""

    milvus_settings = from_context(provides=MilvusSettings, scope=Scope.APP)

    @provide(scope=Scope.APP)
    def milvus_db(self, milvus_settings: MilvusSettings, logger: AISearchLogger) -> MilvusDatabase:
        """Получение MilvusDatabase."""
        return MilvusDatabase(settings=milvus_settings, logger=logger)


class LoggerProvider(Provider):
    """Провайдер для логгера."""

    @provide(scope=Scope.APP)
    def get_logger(self, logger_type: LoggerType) -> AISearchLogger:
        """Получение логгера."""
        return AISearchLogger(logger_type=logger_type)


class RedisProvider(Provider):
    """Провайдер для Redis."""

    redis_settings = from_context(provides=RedisSettings, scope=Scope.APP)

    @provide(scope=Scope.APP)
    def redis_client(self, redis_settings: RedisSettings) -> redis.Redis:
        """Получение клиента Redis."""
        dsn = str(redis_settings.dsn)
        connection = redis.from_url(dsn)  # type: ignore
        return connection.client()
