from dishka import Provider, Scope, from_context, provide

from app.common.logger import AISearchLogger
from app.infrastructure.storages.milvus import MilvusDatabase
from app.settings.config import MilvusSettings


class MilvusProvider(Provider):
    """Провайдер для MilvusDB."""

    milvus_settings = from_context(provides=MilvusSettings, scope=Scope.APP)

    @provide(scope=Scope.APP)
    def milvus_db(
        self, milvus_settings: MilvusSettings, logger: AISearchLogger
    ) -> MilvusDatabase:
        """Получение MilvusDatabase."""
        return MilvusDatabase(settings=milvus_settings, logger=logger)
