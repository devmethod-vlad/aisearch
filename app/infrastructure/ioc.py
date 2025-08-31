from dishka import Provider, Scope, from_context, provide

from app.common.logger import AISearchLogger
from app.common.storages.interfaces import KeyValueStorageProtocol
from app.common.storages.redis import RedisStorage
from app.domain.adapters.confluence import ConfluenceAdapter
from app.domain.adapters.edu import EduAdapter
from app.domain.adapters.google import GoogleTablesAdapter
from app.domain.adapters.interfaces import IConfluenceAdapter, IEduAdapter, IGoogleTablesAdapter
from app.infrastructure.storages.interfaces import IVectorDatabase
from app.infrastructure.storages.milvus import MilvusDatabase
from app.services.interfaces import (
    IKnowledgeBaseService,
    ISemanticSearchService,
    ITaskManagerService,
)
from app.services.knowledge_base import KnowledgeBaseService
from app.services.semantic_search import SemanticSearchService
from app.services.taskmanager import TaskManagerService
from app.settings.config import (
    AppSettings,
    MilvusSettings,
    RestrictionSettings,
)


class ApplicationProvider(Provider):
    """Провайдер зависимостей."""

    app_config = from_context(provides=AppSettings, scope=Scope.APP)
    milvus_config = from_context(provides=MilvusSettings, scope=Scope.APP)
    restrictions_config = from_context(provides=RestrictionSettings, scope=Scope.APP)

    logger = provide(AISearchLogger, scope=Scope.APP)
    redis_storage = provide(RedisStorage, scope=Scope.APP, provides=KeyValueStorageProtocol)
    milvus_database = provide(MilvusDatabase, scope=Scope.REQUEST, provides=IVectorDatabase)
    confluence_adapter = provide(
        ConfluenceAdapter, scope=Scope.REQUEST, provides=IConfluenceAdapter
    )
    google_tables_adapter = provide(
        GoogleTablesAdapter, scope=Scope.REQUEST, provides=IGoogleTablesAdapter
    )
    edu_adapter = provide(EduAdapter, scope=Scope.REQUEST, provides=IEduAdapter)
    semantic_search_service = provide(
        SemanticSearchService, scope=Scope.REQUEST, provides=ISemanticSearchService
    )
    taskmanager_service = provide(
        TaskManagerService, scope=Scope.REQUEST, provides=ITaskManagerService
    )
    knowledge_base_service = provide(
        KnowledgeBaseService, scope=Scope.REQUEST, provides=IKnowledgeBaseService
    )
