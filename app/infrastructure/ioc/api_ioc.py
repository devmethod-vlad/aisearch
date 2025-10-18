from app.services.hybrid_search_service import HybridSearchService
from app.services.interfaces import ITaskManagerService, IHybridSearchService
from app.services.taskmanager import TaskManagerService
from dishka import Provider, from_context, Scope, provide

from app.common.logger import AISearchLogger
from app.common.storages.interfaces import KeyValueStorageProtocol
from app.common.storages.redis import RedisStorage
from app.infrastructure.adapters.light_interfaces import ILLMQueue
from app.infrastructure.adapters.queue import LLMQueue

from app.settings.config import Settings, AppSettings


class ApiSlimProvider(Provider):
    """Легкий API - app провайдер"""
    settings = from_context(Settings, scope=Scope.APP)
    app_config = from_context(provides=AppSettings, scope=Scope.APP)
    logger = provide(AISearchLogger, scope=Scope.APP)
    redis_storage = provide(RedisStorage, scope=Scope.APP, provides=KeyValueStorageProtocol)
    queue = provide(LLMQueue, scope=Scope.APP, provides=ILLMQueue)
    taskmanager_service = provide(
        TaskManagerService, scope=Scope.REQUEST, provides=ITaskManagerService
    )
    hybrid_search_service = provide(
        HybridSearchService, scope=Scope.REQUEST, provides=IHybridSearchService
    )
