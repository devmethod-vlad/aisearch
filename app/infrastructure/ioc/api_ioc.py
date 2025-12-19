from dishka import Provider, Scope, from_context, provide

from app.common.database import Database
from app.common.logger import AISearchLogger
from app.common.storages.interfaces import KeyValueStorageProtocol
from app.common.storages.redis import RedisStorage
from app.infrastructure.adapters.light_interfaces import ILLMQueue
from app.infrastructure.adapters.queue import LLMQueue
from app.infrastructure.healthchecks.milvus_check import MilvusHealthCheck
from app.infrastructure.healthchecks.opensearch_check import OpenSearchHealthCheck
from app.infrastructure.healthchecks.postgres_check import PostgresHealthCheck
from app.infrastructure.healthchecks.redis_check import RedisHealthCheck
from app.infrastructure.unit_of_work.interfaces import IUnitOfWork
from app.infrastructure.unit_of_work.uow import UnitOfWork
from app.services.feedback import FeedbackService
from app.services.hybrid_search_service import HybridSearchService
from app.services.interfaces import (
    IFeedbackService,
    IHybridSearchService,
    ITaskManagerService,
)
from app.services.taskmanager import TaskManagerService
from app.settings.config import AppSettings, PostgresSettings, Settings


class ApiSlimProvider(Provider):
    """Легкий API - app провайдер"""

    settings = from_context(Settings, scope=Scope.APP)
    app_config = from_context(provides=AppSettings, scope=Scope.APP)
    postgres_config = from_context(provides=PostgresSettings, scope=Scope.APP)
    logger = provide(AISearchLogger, scope=Scope.APP)
    redis_storage = provide(
        RedisStorage, scope=Scope.APP, provides=KeyValueStorageProtocol
    )
    queue = provide(LLMQueue, scope=Scope.APP, provides=ILLMQueue)
    database = provide(Database, scope=Scope.APP)
    taskmanager_service = provide(
        TaskManagerService, scope=Scope.REQUEST, provides=ITaskManagerService
    )
    hybrid_search_service = provide(
        HybridSearchService, scope=Scope.REQUEST, provides=IHybridSearchService
    )
    unit_of_work = provide(UnitOfWork, scope=Scope.REQUEST, provides=IUnitOfWork)
    feedback_service = provide(
        FeedbackService, scope=Scope.REQUEST, provides=IFeedbackService
    )


class HealthCheckProvider(Provider):
    """Провайдер для health checks"""

    @provide(scope=Scope.APP)
    def provide_postgres_health_check(self, database: Database) -> PostgresHealthCheck:
        return PostgresHealthCheck(database)

    @provide(scope=Scope.APP)
    def provide_redis_health_check(
        self, redis_storage: KeyValueStorageProtocol
    ) -> RedisHealthCheck:
        return RedisHealthCheck(redis_storage)

    @provide(scope=Scope.APP)
    def provide_opensearch_health_check(self) -> OpenSearchHealthCheck:
        return OpenSearchHealthCheck()

    @provide(scope=Scope.APP)
    def provide_milvus_health_check(self) -> MilvusHealthCheck:
        return MilvusHealthCheck()
