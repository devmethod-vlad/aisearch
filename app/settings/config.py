import typing as tp
from collections.abc import Sequence
from datetime import time
from pathlib import Path

from pydantic import (
    PostgresDsn,
    RedisDsn,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvBaseSettings(BaseSettings):
    """Базовый класс для прокидывания настроек из .env."""

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
    use_cache: bool = True

    logs_path: str | None = None
    logs_access_path: str | None = None
    log_level: str = "INFO"

    normalize_query: bool
    collection_file_path: str

    model_config = SettingsConfigDict(env_prefix="app_")


class MilvusSettings(EnvBaseSettings):
    """Настройки MilvusDB."""

    host: str
    port: int
    web_ui_port: int
    use_ssl: bool = False
    connection_timeout: int
    query_timeout: int
    collection_name: str
    model_name: str
    recreate_collection: bool = False
    vector_field: str = "embedding"
    schema_path: str = "app/settings/conf.json"
    id_field: str = "ext_id"
    metric_type: str = "IP"
    search_fields: str = "question"
    output_fields: str | list[str] = (
        "row_idx,source,ext_id,page_id,role,component,question,analysis,answer"
    )

    @model_validator(mode="after")
    def assemble_milvus_settings(self) -> tp.Self:
        """Досборка настроек MilvusDB"""
        if isinstance(self.output_fields, str):
            self.output_fields = self.output_fields.split(",")
        return self

    model_config = SettingsConfigDict(env_prefix="milvus_")


class RedisSettings(EnvBaseSettings):
    """Настройки Redis"""

    hostname: str
    port: int
    database: int
    dsn: RedisDsn | str | None = None

    @model_validator(mode="after")
    def assemble_redis_connection(self) -> tp.Self:
        """Сборка Redis DSN"""
        if self.dsn is None:
            self.dsn = str(
                RedisDsn.build(
                    scheme="redis",
                    host=self.hostname,
                    port=self.port,
                    path=f"/{self.database}",
                )
            )
        return self

    model_config = SettingsConfigDict(env_prefix="redis_")


class VLLMSettings(EnvBaseSettings):
    """Настройки клиента LLM"""

    base_url: str
    port: int
    api_key: str | None = None
    model: str | None = None
    model_name: str
    max_input_tokens: int = 4096
    max_output_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    request_timeout: int = 60
    stream: bool = False
    model_config = SettingsConfigDict(env_prefix="vllm_")


class HybridSearchSettings(EnvBaseSettings):
    """Настройки глобального поиска"""

    dense_top_k: int = 20
    lex_top_k: int = 50
    top_k: int = 5
    w_ce: float = 0.6
    w_dense: float = 0.25
    w_lex: float = 0.15
    dense_abs_min: float = 0.25
    dense_rel_min: float = 0.6
    lex_rel_min: float = 0.5
    precut_min_keep: int = 3
    cache_ttl: int = 3600
    version: str = "v1"
    collection_name: str = "kb_default"
    merge_by_field: str = "ext_id"
    merge_fields: str | list[str] = (
        "row_idx,source,ext_id,page_id,role,component,question,analysis,answer"
    )
    enable_intermediate_results: bool
    intermediate_results_top_k: int

    @model_validator(mode="after")
    def assemble_hybrid_settings(self) -> tp.Self:
        """Парсинг merge_fields из строки в список"""
        if isinstance(self.merge_fields, str):
            self.merge_fields = [
                f.strip() for f in self.merge_fields.split(",") if f.strip()
            ]

        return self

    model_config = SettingsConfigDict(env_prefix="hybrid_")


class SearchSwitches(EnvBaseSettings):
    """Настройки переключателей поиска"""

    use_opensearch: bool
    use_reranker: bool
    use_hybrid: bool
    model_config = SettingsConfigDict(env_prefix="search_")


class LLMQueueSettings(EnvBaseSettings):
    """Настройки очереди вне семафора"""

    queue_list_key: str = "llm:queue:list"
    ticket_hash_prefix: str = "llm:ticket:"
    max_size: int = 100
    ticket_ttl: int = 3600
    drain_interval_sec: int = 1
    processing_list_key: str | None = None

    model_config = SettingsConfigDict(env_prefix="llm_queue_")


class LLMGlobalSemaphoreSettings(EnvBaseSettings):
    """Настройки глобального семафора"""

    key: str = "llm:{global}:sem"
    limit: int = 2
    ttl_ms: int = 120000
    wait_timeout_ms: int = 1
    heartbeat_ms: int = 30000

    model_config = SettingsConfigDict(env_prefix="llm_global_sem_")


class OpenSearchSettings(EnvBaseSettings):
    """Настройки OpenSearch"""

    host: str
    port: int
    index_name: str
    use_ssl: bool = False
    verify_certs: bool = False
    user: str | None = None
    password: str | None = None
    search_fields: str | list[str] = "question,analysis,answer"
    output_fields: str | list[str] = (
        "row_idx,source,ext_id,page_id,role,component,question,analysis,answer"
    )
    operator: str = "or"
    min_should_match: int = 1
    fuzziness: int = 0
    use_rescore: bool = False
    index_answer: bool = True
    schema_path: str = "app/settings/os_index.json"
    bulk_chunk_size: int = 1000
    recreate_index: bool = True
    model_config = SettingsConfigDict(env_prefix="os_")

    @model_validator(mode="after")
    def assemble_os_settings(self) -> tp.Self:
        """Парсинг fields из строки в список"""
        if isinstance(self.search_fields, str):
            self.search_fields = [
                f.strip() for f in self.search_fields.split(",") if f.strip()
            ]
        if isinstance(self.output_fields, str):
            self.output_fields = [
                f.strip() for f in self.output_fields.split(",") if f.strip()
            ]

        return self


class SlowAPISettings(EnvBaseSettings):
    """Настройки slowapi лимитов"""

    search: str
    generate: str

    model_config = SettingsConfigDict(env_prefix="slowapi_limit_")


class RerankerSettings(EnvBaseSettings):
    """Cross Encoder настройки"""

    model_name: str
    device: str = "cpu"
    batch_size: int = 128
    max_length: int = 192
    score_mode: str = "sigmoid"
    softmax_temp: float = 0.7
    pairs_fields: str | list[str]
    dtype: str = "fp16"
    model_config = SettingsConfigDict(env_prefix="reranker_")

    @model_validator(mode="after")
    def assemble_pairs_settings(self) -> tp.Self:
        """Парсинг pairs_fields из строки в список"""
        if isinstance(self.pairs_fields, str):
            self.pairs_fields = [
                f.strip() for f in self.pairs_fields.split(",") if f.strip()
            ]

        return self


class WarmupSettings(BaseSettings):
    """Прогрев модели"""

    enabled: bool = True
    ce_pairs: str = ""
    embed_texts: str = ""
    model_config = SettingsConfigDict(env_prefix="warmup_")


class CelerySettings(EnvBaseSettings):
    """Настройки Celery"""

    logs_path: str | None = None
    log_level: str = "INFO"
    logs_queue_path: str | None = None
    log_queue_level: str = "INFO"

    workers_num: int

    model_config = SettingsConfigDict(env_prefix="celery_")


class ExtractEduSettings(EnvBaseSettings):
    """Извлечение файлов с edu"""

    edu_emias_url: str
    base_harvester_api_url: str
    edu_emias_token: str
    edu_emias_attachments_page_id: str
    edu_timeout: int
    knowledge_base_file_name: str
    vio_base_file_name: str
    cron_update_times: str
    vio_harvester_suffix: str
    kb_harvester_suffix: str

    logs_path: str | None = None
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_prefix="extract_")

    @field_validator("cron_update_times")
    @classmethod
    def validate_cron_hours(cls, v: str) -> str:
        """Валидация cron_update_times"""
        if not v:
            raise ValueError("cron_update_times не может быть пустой строкой")

        for time_str in v.split(","):
            time_s = time_str.strip()
            try:
                hour_str, minute_str = time_s.split(":")
                time(int(hour_str), int(minute_str))
            except Exception:
                raise ValueError(
                    f'Некорректное время "{time_s}". '
                    f'Используйте формат HH:MM, например "09:30" или "13:20, 15:00"'
                )

        return v


class ShortSettings(EnvBaseSettings):
    """Настройки для short запросов"""

    mode: bool = True
    mode_limit: int
    use_opensearch: bool
    use_reranker: bool
    use_hybrid: bool
    dense_top_k: int = 20
    lex_top_k: int = 50
    top_k: int = 5
    w_ce: float = 0.6
    w_dense: float = 0.25
    w_lex: float = 0.15

    model_config = SettingsConfigDict(env_prefix="short_")


class SearchMetricsEnabled(EnvBaseSettings):
    """Вывод временных метрик поиска"""

    log_metrics_enabled: bool = True
    response_metrics_enabled: bool = True

    model_config = SettingsConfigDict(env_prefix="timeit_")


class PostgresSettings(EnvBaseSettings):
    """Настройки Postgres"""

    engine: str = "postgresql"
    host: str
    port: int
    user: str
    password: str
    db: str
    pool_size: int | None = None
    pool_overflow_size: int | None = None
    leader_usage_coefficient: float | None = None
    use_async: bool = True
    echo: bool = False
    autoflush: bool = False
    autocommit: bool = False
    expire_on_commit: bool = False
    engine_health_check_delay: int | None = None
    dsn: PostgresDsn | None = None
    slave_hosts: Sequence[str] | str = ""
    slave_dsns: Sequence[PostgresDsn] | str = ""

    @model_validator(mode="after")
    def assemble_db_connection(self) -> tp.Self:
        """Сборка Postgres DSN"""
        if self.dsn is None:
            self.dsn = str(
                PostgresDsn.build(
                    scheme=self.engine + "+asyncpg" if self.use_async else "",
                    username=self.user,
                    password=self.password,
                    host=self.host,
                    port=self.port,
                    path=f"{self.db}",
                )
            )
        return self

    model_config = SettingsConfigDict(env_prefix="postgres_")


class Settings(EnvBaseSettings):
    """Настройки проекта."""

    app: AppSettings = AppSettings()
    database: PostgresSettings = PostgresSettings()
    milvus: MilvusSettings = MilvusSettings()
    redis: RedisSettings = RedisSettings()
    celery: CelerySettings = CelerySettings()
    vllm: VLLMSettings = VLLMSettings()
    hybrid: HybridSearchSettings = HybridSearchSettings()
    llm_queue: LLMQueueSettings = LLMQueueSettings()
    llm_global_sem: LLMGlobalSemaphoreSettings = LLMGlobalSemaphoreSettings()
    opensearch: OpenSearchSettings = OpenSearchSettings()
    reranker: RerankerSettings = RerankerSettings()
    warmup: WarmupSettings = WarmupSettings()
    switches: SearchSwitches = SearchSwitches()
    slowapi: SlowAPISettings = SlowAPISettings()
    search_metrics: SearchMetricsEnabled = SearchMetricsEnabled()
    extract_edu: ExtractEduSettings = ExtractEduSettings()
    postgres: PostgresSettings = PostgresSettings()
    short_settings: ShortSettings = ShortSettings()

    @model_validator(mode="after")
    def _fill_vllm_model(self) -> tp.Self:
        if not (getattr(self.vllm, "model", None)):
            base = getattr(self.app, "modelstore_contr_path", None)
            name = getattr(self.vllm, "model_name", None)
            if base and name:
                self.vllm.model = str(Path(base) / name)
        return self


settings = Settings()
