from dishka import Provider, Scope, from_context, provide

from app.common.logger import AISearchLogger
from app.common.storages.interfaces import KeyValueStorageProtocol
from app.common.storages.redis import RedisStorage
from app.infrastructure.adapters.light_interfaces import ILLMQueue
from app.infrastructure.adapters.queue import LLMQueue
from app.settings.config import AppSettings, Settings


class QueueSlimProvider(Provider):
    """Легкий Queue провайдер"""

    settings = from_context(Settings, scope=Scope.APP)
    app_config = from_context(provides=AppSettings, scope=Scope.APP)
    logger = provide(AISearchLogger, scope=Scope.APP)
    redis_storage = provide(
        RedisStorage, scope=Scope.APP, provides=KeyValueStorageProtocol
    )
    queue = provide(LLMQueue, scope=Scope.APP, provides=ILLMQueue)
