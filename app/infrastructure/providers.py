import redis.asyncio as redis
from dishka import Provider, Scope, from_context, provide

from app.common.logger import AISearchLogger, LoggerType

from app.settings.config import RedisSettings



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
