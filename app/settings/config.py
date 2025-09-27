import typing as tp
from pathlib import Path
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

    model_config = SettingsConfigDict(env_prefix="app_")


class MilvusSettings(EnvBaseSettings):
    """Настройки MilvusDB."""

    host: str
    port: int
    use_ssl: bool = False
    connection_timeout: int
    query_timeout: int
    preloaded_collection_names: str | list[str]
    model_name: str
    recreate_collection: bool = False
    vector_field: str = "embedding"
    id_field: str = "ext_id"
    output_fields: str | list[str] = (
        "row_idx,source,ext_id,page_id,role,component,question,analysis,answer"
    )

    @model_validator(mode="after")
    def assemble_milvus_settings(self) -> tp.Self:
        """Досборка настроек MilvusDB"""
        if isinstance(self.preloaded_collection_names, str):
            self.preloaded_collection_names = self.preloaded_collection_names.split(",")
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
    dense_threshold: float = 0.0
    lex_threshold: float = 0.0
    ce_threshold: float = 0.0
    cache_ttl: int = 3600
    version: str = "v1"
    collection_name: str = "kb_default"
    model_config = SettingsConfigDict(env_prefix="hybrid_")


class SearchSwitches(EnvBaseSettings):
    """Настройки переключателей поиска"""

    use_opensearch: bool
    use_bm25: bool
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
    # query_profile: str = "fast"
    # query_fields: str = "question,analysis,answer"
    operator: str = "or"
    min_should_match: int = 1
    fuzziness: int = 0
    use_rescore: bool = False
    index_answer: bool = True
    bulk_chunk_size: int = 1000
    recreate_index: bool = True
    model_config = SettingsConfigDict(env_prefix="os_")


class BM25Settings(EnvBaseSettings):
    """Настройки BM25"""

    engine: str = "whoosh"
    index_path: str = "/data/bm25_index"
    schema_fields: str = "question,analysis,answer"
    recreate_index: bool = False
    model_config = SettingsConfigDict(env_prefix="bm25_")


class RerankerSettings(EnvBaseSettings):
    """Cross Encoder настройки"""

    model_name: str
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


class Settings(EnvBaseSettings):
    """Настройки проекта."""

    app: AppSettings = AppSettings()
    milvus: MilvusSettings = MilvusSettings()
    redis: RedisSettings = RedisSettings()
    celery: CelerySettings = CelerySettings()
    vllm: VLLMSettings = VLLMSettings()
    hybrid: HybridSearchSettings = HybridSearchSettings()
    llm_queue: LLMQueueSettings = LLMQueueSettings()
    llm_global_sem: LLMGlobalSemaphoreSettings = LLMGlobalSemaphoreSettings()
    opensearch: OpenSearchSettings = OpenSearchSettings()
    bm25: BM25Settings = BM25Settings()
    reranker: RerankerSettings = RerankerSettings()
    warmup: WarmupSettings = WarmupSettings()
    switches: SearchSwitches = SearchSwitches()

    @model_validator(mode="after")
    def _fill_vllm_model(self) -> Self:
        if not (getattr(self.vllm, "model", None)):
            base = getattr(self.app, "modelstore_contr_path", None)
            name = getattr(self.vllm, "model_name", None)
            if base and name:
                self.vllm.model = str(Path(base) / name)
        return self


settings = Settings()
