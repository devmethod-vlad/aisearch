import typing as tp
from collections.abc import Sequence
from datetime import time

from pydantic import (
    PostgresDsn,
    RedisDsn,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class EnvBaseSettings(BaseSettings):
    """Базовый класс для прокидывания настроек из .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


def _validate_cron_update_times(v: str) -> str:
    """Валидирует строку расписания в формате HH:MM[,HH:MM]."""
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

    logs_path: str | None = None
    logs_access_path: str | None = None
    log_level: str = "INFO"

    normalize_query: bool
    collection_files_contr_dir: str = "/collections"
    field_mapping_schema_path: str = "app/settings/field_mapping.json"

    data_unique_id: str = "ext_id"
    recreate_data: bool = False

    generate_prelaunch_data: bool = False
    prelaunch_data_contr_dir: str = "/prelaunch"

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
    vector_field: str = "embedding"
    schema_path: str = "app/settings/conf.json"
    id_field: str = "ext_id"
    metric_type: str = "IP"
    search_fields: str = "question"
    output_fields: str | list[str] = (
        "row_idx,source,ext_id,page_id,role,component,question,analysis,answer,answer_copy"
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
        "row_idx,source,ext_id,page_id,role,component,question,analysis,answer,answer_copy"
    )
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


class TokenFiltersSettings(EnvBaseSettings):
    """Настройки token-фильтрации мультизначных полей."""

    env_separator: str = ","
    raw_fields: tp.Annotated[tuple[str, ...], NoDecode] = ()
    token_suffix: str = "_tokens"
    raw_separator: str = ";"

    @field_validator("raw_fields", mode="before")
    @classmethod
    def parse_raw_fields(cls, v: tp.Any, info: ValidationInfo) -> tuple[str, ...]:
        separator = info.data.get("env_separator", ",")

        if isinstance(v, str):
            values = [item.strip() for item in v.split(separator) if item.strip()]
            return tuple(values)

        if isinstance(v, (list, tuple, set)):
            values = [str(item).strip() for item in v if str(item).strip()]
            return tuple(values)

        raise ValueError("raw_fields должен быть строкой или списком значений")

    model_config = SettingsConfigDict(env_prefix="token_filters_")




class ExactFiltersSettings(EnvBaseSettings):
    """Настройки exact-фильтрации single-value полей."""

    env_separator: str = ","
    raw_fields: tp.Annotated[tuple[str, ...], NoDecode] = ()
    field_suffix: str = "_filter"

    @field_validator("raw_fields", mode="before")
    @classmethod
    def parse_raw_fields(cls, v: tp.Any, info: ValidationInfo) -> tuple[str, ...]:
        """Нормализует список полей exact-фильтрации из env/коллекций."""
        separator = info.data.get("env_separator", ",")
        if isinstance(v, str):
            return tuple(item.strip() for item in v.split(separator) if item.strip())
        if isinstance(v, (list, tuple, set)):
            return tuple(str(item).strip() for item in v if str(item).strip())
        raise ValueError("raw_fields должен быть строкой или списком значений")

    model_config = SettingsConfigDict(env_prefix="exact_filters_")


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
        "row_idx,source,ext_id,page_id,role,component,question,analysis,answer,answer_copy"
    )
    operator: str = "or"
    min_should_match: int = 1
    fuzziness: int = 0
    use_rescore: bool = False
    index_answer: bool = True
    schema_path: str = "app/settings/os_index.json"
    bulk_chunk_size: int = 1000
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

    model_config = SettingsConfigDict(env_prefix="slowapi_limit_")


class RerankerSettings(EnvBaseSettings):
    """Cross Encoder настройки"""

    model_name: str
    device: str = "cuda"
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

    deduplicated_excel_upload_enabled: bool = False
    deduplicated_excel_file_name_template: str = "statistic_{timestamp}.xlsx"
    deduplicated_excel_keep_versions: int = 5
    deduplicated_excel_max_retries: int = 5
    deduplicated_excel_retry_backoff_base_seconds: float = 1.0
    deduplicated_excel_retry_backoff_max_seconds: float = 30.0

    logs_path: str | None = None
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_prefix="extract_")

    @field_validator("cron_update_times")
    @classmethod
    def validate_cron_hours(cls, v: str) -> str:
        """Валидация cron_update_times"""
        return _validate_cron_update_times(v)

    @field_validator("deduplicated_excel_keep_versions")
    @classmethod
    def validate_deduplicated_excel_keep_versions(cls, value: int) -> int:
        """Проверяет, что ограничение числа версий attachment >= 1."""
        if value < 1:
            raise ValueError("deduplicated_excel_keep_versions должен быть >= 1")
        return value

    @field_validator("deduplicated_excel_max_retries")
    @classmethod
    def validate_deduplicated_excel_max_retries(cls, value: int) -> int:
        """Проверяет, что число попыток запросов >= 1."""
        if value < 1:
            raise ValueError("deduplicated_excel_max_retries должен быть >= 1")
        return value

    @field_validator(
        "deduplicated_excel_retry_backoff_base_seconds",
        "deduplicated_excel_retry_backoff_max_seconds",
    )
    @classmethod
    def validate_deduplicated_excel_backoff_values(cls, value: float) -> float:
        """Проверяет, что значения backoff не отрицательные."""
        if value < 0:
            raise ValueError("значения backoff не могут быть отрицательными")
        return value


class GlossarySettings(EnvBaseSettings):
    """Настройки синхронизации глоссария аббревиатур."""

    api_url: str = (
        "https://edu.emias.ru/edu-rest-api/test/glossary/glossary/getelements"
    )
    page_limit: int = 500
    cron_update_times: str
    request_timeout: int = 30
    max_retries: int = 5
    retry_backoff_base_seconds: float = 1.0
    retry_backoff_max_seconds: float = 30.0
    abbreviation_delimiter: str = ";"
    term_delimiter: str = ";"

    model_config = SettingsConfigDict(env_prefix="glossary_")

    @field_validator("page_limit")
    @classmethod
    def validate_page_limit(cls, value: int) -> int:
        """Проверяет допустимый размер страницы для внешнего API."""
        if value < 1 or value > 500:
            raise ValueError("page_limit должен быть в диапазоне от 1 до 500")
        return value

    @field_validator("cron_update_times")
    @classmethod
    def validate_cron_hours(cls, value: str) -> str:
        """Валидирует расписание запуска синхронизации глоссария."""
        return _validate_cron_update_times(value)


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
    hybrid: HybridSearchSettings = HybridSearchSettings()
    token_filters: TokenFiltersSettings = TokenFiltersSettings()
    exact_filters: ExactFiltersSettings = ExactFiltersSettings()
    llm_queue: LLMQueueSettings = LLMQueueSettings()
    llm_global_sem: LLMGlobalSemaphoreSettings = LLMGlobalSemaphoreSettings()
    opensearch: OpenSearchSettings = OpenSearchSettings()
    reranker: RerankerSettings = RerankerSettings()
    warmup: WarmupSettings = WarmupSettings()
    switches: SearchSwitches = SearchSwitches()
    slowapi: SlowAPISettings = SlowAPISettings()
    search_metrics: SearchMetricsEnabled = SearchMetricsEnabled()
    extract_edu: ExtractEduSettings = ExtractEduSettings()
    glossary: GlossarySettings = GlossarySettings()
    postgres: PostgresSettings = PostgresSettings()
    short_settings: ShortSettings = ShortSettings()


settings = Settings()
