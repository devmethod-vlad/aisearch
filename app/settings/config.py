import datetime
from typing import Self

from pydantic import RedisDsn, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvBaseSettings(BaseSettings):
    """Базовый класс для прокидывания настроек из env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class AppSettings(EnvBaseSettings):
    """Настройки приложения FastAPI."""

    mode: str = "dev"
    host: str
    port: int
    debug_host: str
    debug_port: int
    workers_num: int
    access_key: str
    prefix: str = ""

    logs_host_path: str
    logs_contr_path: str

    model_name: str

    knowledge_base_minitable_google_link: str
    knowledge_base_megatable_google_link: str
    knowledge_base_main_header_text: str
    knowledge_base_target_columns: str | list[str]
    knowledge_base_roles_separator: str
    knowledge_base_collect_data_time: datetime.time

    confluence_url: str
    confluence_token: str

    edu_emias_url: str
    edu_emias_token: str
    edu_emias_attachments_page_id: str

    @model_validator(mode="after")
    def assemble_app_settings(self) -> Self:
        """Досборка настроек приложения"""
        if isinstance(self.knowledge_base_target_columns, str):
            self.knowledge_base_target_columns = self.knowledge_base_target_columns.split(";")
        return self

    model_config = SettingsConfigDict(env_prefix="app_")


class MilvusSettings(EnvBaseSettings):
    """Настройки MilvusDB."""

    host: str
    port: int
    nlist: int
    nprobe: int
    load_timeout: int
    query_timeout: int

    model_config = SettingsConfigDict(env_prefix="milvus_")


class RedisSettings(EnvBaseSettings):
    """Настройки Redis"""

    hostname: str
    port: int
    database: int
    dsn: RedisDsn | str | None = None

    @model_validator(mode="after")
    def assemble_redis_connection(self) -> Self:
        """Сборка Redis DSN"""
        if self.dsn is None:
            self.dsn = str(
                RedisDsn.build(
                    scheme="redis", host=self.hostname, port=self.port, path=f"/{self.database}"
                )
            )
        return self

    model_config = SettingsConfigDict(env_prefix="redis_")


class RestrictionSettings(EnvBaseSettings):
    """Настройки ограничителей запросов, очередей, кэша"""

    queue_key: str = "celery:pending-tasks-ids"
    base_cache_key: str = "celery:cache-result"
    max_cache_ttl: int

    semantic_search_last_query_time_key: str = "celery:semantic-search:last-query-time"
    semantic_search_timeout_interval: int
    semantic_search_queue_size: int

    model_config = SettingsConfigDict(env_prefix="restrict_")


class CelerySettings(EnvBaseSettings):
    """Настройки Celery"""

    logs_host_path: str
    logs_contr_path: str
    workers_num: int

    model_config = SettingsConfigDict(env_prefix="celery_")


class Settings(EnvBaseSettings):
    """Настройки проекта."""

    app: AppSettings = AppSettings()
    milvus: MilvusSettings = MilvusSettings()
    redis: RedisSettings = RedisSettings()
    restrictions: RestrictionSettings = RestrictionSettings()
    celery: CelerySettings = CelerySettings()


settings = Settings()
