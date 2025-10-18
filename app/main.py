

from dishka import make_async_container
from dishka.integrations.fastapi import FastapiProvider, setup_dishka
from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIASGIMiddleware

from app.api.v1.routers.base import router as base_router
from app.api.v1.routers.hybrid_search import router as hybrid_router, limiter
from app.api.v1.routers.taskmanager import router as taskmanager_router
from app.common.logger import LoggerType
from app.common.providers import AuthProvider
from app.domain.exception_handler import exception_config
from app.infrastructure.ioc.api_ioc import ApiSlimProvider

from app.infrastructure.providers import (
    LoggerProvider,
    RedisProvider,
)


from app.settings.config import (
    AppSettings,
    RedisSettings,
    Settings,
    settings,
)


def create_app() -> FastAPI:
    """Инициализация приложения"""
    application = FastAPI(title="AI Search", root_path=settings.app.prefix)
    container = make_async_container(
        ApiSlimProvider(),
        FastapiProvider(),
        AuthProvider(),
        LoggerProvider(),
        RedisProvider(),
        context={
            AppSettings: settings.app,
            RedisSettings: settings.redis,
            LoggerType: LoggerType.APP,
            Settings: settings,
        },
    )
    setup_dishka(container, application)
    application.include_router(base_router)
    application.include_router(taskmanager_router)
    application.include_router(hybrid_router)
    application.state.limiter = limiter
    application.add_middleware(SlowAPIASGIMiddleware)

    for exception, handler in exception_config.items():
        application.add_exception_handler(exception, handler)
    application.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    return application


app = create_app()
