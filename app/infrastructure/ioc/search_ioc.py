from dishka import Provider, Scope, from_context, provide

from app.common.logger import AISearchLogger
from app.common.storages.interfaces import KeyValueStorageProtocol
from app.common.storages.redis import RedisStorage
from app.infrastructure.adapters.bm25 import BM25Adapter
from app.infrastructure.adapters.cross_encoder import CrossEncoderAdapter
from app.infrastructure.adapters.edu import EduAdapter
from app.infrastructure.adapters.interfaces import (
    IBM25Adapter,
    ICrossEncoderAdapter,
    IOpenSearchAdapter,
    IRedisSemaphore,
    IVLLMAdapter, IEduAdapter,
)
from app.infrastructure.adapters.light_interfaces import ILLMQueue
from app.infrastructure.adapters.llm_adapter import VLLMAdapter
from app.infrastructure.adapters.open_search import OpenSearchAdapter
from app.infrastructure.adapters.queue import LLMQueue
from app.infrastructure.adapters.semaphore import RedisSemaphore
from app.infrastructure.storages.interfaces import IVectorDatabase
from app.infrastructure.storages.milvus import MilvusDatabase
from app.services.hybrid_search_orchestrator import HybridSearchOrchestrator

from app.services.interfaces import (
    IHybridSearchOrchestrator, IUpdaterService,

)
from app.services.updater import UpdaterService

from app.settings.config import (
    AppSettings,
    MilvusSettings,
    Settings,
)


class ApplicationProvider(Provider):
    """Провайдер зависимостей."""

    settings = from_context(Settings, scope=Scope.APP)
    app_config = from_context(provides=AppSettings, scope=Scope.APP)
    milvus_config = from_context(provides=MilvusSettings, scope=Scope.APP)

    logger = provide(AISearchLogger, scope=Scope.APP)
    redis_storage = provide(RedisStorage, scope=Scope.APP, provides=KeyValueStorageProtocol)
    redis_semaphore = provide(RedisSemaphore, scope=Scope.APP, provides=IRedisSemaphore)
    queue = provide(LLMQueue, scope=Scope.APP, provides=ILLMQueue)
    vllm_client = provide(VLLMAdapter, scope=Scope.APP, provides=IVLLMAdapter)
    os_adapter = provide(OpenSearchAdapter, scope=Scope.APP, provides=IOpenSearchAdapter)
    bm25_adapter = provide(BM25Adapter, scope=Scope.APP, provides=IBM25Adapter)
    cross_encoder_adapter = provide(
        CrossEncoderAdapter, scope=Scope.APP, provides=ICrossEncoderAdapter
    )
    edu_adapter = provide(EduAdapter, scope=Scope.APP, provides=IEduAdapter)
    update_service = provide(UpdaterService, scope=Scope.APP, provides=IUpdaterService)
    milvus_database = provide(MilvusDatabase, scope=Scope.APP, provides=IVectorDatabase)

    hybrid_search_orchestrator = provide(
        HybridSearchOrchestrator, scope=Scope.APP, provides=IHybridSearchOrchestrator
    )

