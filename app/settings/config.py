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


class VLLMSettings(EnvBaseSettings):
    """Настройки клиента LLM"""

    base_url: str
    api_key: str | None = None
    model: str
    max_input_tokens: int = 4096
    max_output_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    request_timeout: int = 60
    stream: bool = False
    model_config = SettingsConfigDict(env_prefix="vllm_")


class ConcurrencySettings(EnvBaseSettings):
    """Настройки параллельности локально"""

    local_llm_sem: int = 2
    model_config = SettingsConfigDict(env_prefix="concurrency_")


class HybridSearchSettings(EnvBaseSettings):
    """Настройки глобального поиска"""

    dense_top_k: int = 20
    lex_top_k: int = 50
    top_k: int = 5
    w_ce: float = 0.6
    w_dense: float = 0.25
    w_lex: float = 0.15
    dense_threshold: float = 0.0
    lex_threshold: float = 0.0
    ce_threshold: float = 0.0
    cache_ttl: int = 3600
    version: str = "v1"
    model_config = SettingsConfigDict(env_prefix="hybrid_")


class SearchSwitches(EnvBaseSettings):
    """Настройки переключателей поиска"""

    use_opensearch: bool = True
    use_bm25: bool = False
    use_reranker: bool = True
    use_hybrid: bool = True  # общий выключатель гибрида
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
    wait_timeout_ms: int = 30000
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
    query_profile: str = "fast"
    query_fields: str = "question,analysis,answer"
    operator: str = "or"
    min_should_match: int = 1
    fuzziness: int = 0
    use_rescore: bool = False
    model_config = SettingsConfigDict(env_prefix="os_")


class BM25Settings(EnvBaseSettings):
    """Настройки BM25"""

    engine: str = "whoosh"
    index_path: str = "/data/bm25_index"
    schema_fields: str = "question,analysis,answer"
    model_config = SettingsConfigDict(env_prefix="bm25_")


class RerankerSettings(EnvBaseSettings):
    """Cross Encoder настройки"""

    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cpu"
    model_config = SettingsConfigDict(env_prefix="reranker_")


class WarmupSettings(BaseSettings):
    """Прогрев модели"""

    enabled: bool = True
    ce_pairs: str = ""
    embed_texts: str = ""
    model_config = SettingsConfigDict(env_prefix="warmup_")


class CelerySettings(EnvBaseSettings):
    """Настройки Celery"""

    logs_host_path: str
    logs_contr_path: str
    workers_num: int

    model_config = SettingsConfigDict(env_prefix="celery_")


class MilvusDenseSettings(EnvBaseSettings):
    """Настройки Milvus"""

    host: str = "aisearch-milvus"
    port: int = 19530
    collection: str = "kb_default"
    vector_field: str = "embedding"
    id_field: str = "ext_id"
    output_fields: str = "ext_id,question,analysis,answer"
    app_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model_config = SettingsConfigDict(env_prefix="milvus_")


class Settings(EnvBaseSettings):
    """Настройки проекта."""

    app: AppSettings = AppSettings()
    milvus: MilvusSettings = MilvusSettings()
    redis: RedisSettings = RedisSettings()
    restrictions: RestrictionSettings = RestrictionSettings()
    celery: CelerySettings = CelerySettings()
    vllm: VLLMSettings = VLLMSettings()
    hybrid: HybridSearchSettings = HybridSearchSettings()
    search: SearchSwitches = SearchSwitches()
    llm_queue: LLMQueueSettings = LLMQueueSettings()
    llm_global_sem: LLMGlobalSemaphoreSettings = LLMGlobalSemaphoreSettings()
    opensearch: OpenSearchSettings = OpenSearchSettings()
    bm25: BM25Settings = BM25Settings()
    reranker: RerankerSettings = RerankerSettings()
    milvus_dense: MilvusDenseSettings = MilvusDenseSettings()
    warmup: WarmupSettings = WarmupSettings()


settings = Settings()
