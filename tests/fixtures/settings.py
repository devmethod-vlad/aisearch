import os
from pathlib import Path

import pytest
from testcontainers.milvus import MilvusContainer
from testcontainers.opensearch import OpenSearchContainer

from app.settings.config import (
    AppSettings,
    CelerySettings,
    ExtractEduSettings,
    HybridSearchSettings,
    LLMGlobalSemaphoreSettings,
    LLMQueueSettings,
    MilvusSettings,
    OpenSearchSettings,
    PostgresSettings,
    RedisSettings,
    RerankerSettings,
    SearchMetricsEnabled,
    SearchSwitches,
    Settings,
    ShortSettings,
    SlowAPISettings,
    VLLMSettings,
    WarmupSettings,
)


@pytest.fixture(scope="session")
def test_settings(
    test_app_settings: AppSettings,
    test_milvus_settings: MilvusSettings,
    test_opensearch_settings: OpenSearchSettings,
    test_database_settings: PostgresSettings,
    test_redis_settings: RedisSettings,
    test_celery_settings: CelerySettings,
    test_vllm_settings: VLLMSettings,
    test_hybrid_settings: HybridSearchSettings,
    test_llm_queue_settings: LLMQueueSettings,
    test_llm_global_sem_settings: LLMGlobalSemaphoreSettings,
    test_reranker_settings: RerankerSettings,
    test_warmup_settings: WarmupSettings,
    test_switches_settings: SearchSwitches,
    test_slowapi_settings: SlowAPISettings,
    test_search_metrics_settings: SearchMetricsEnabled,
    test_extract_edu_settings: ExtractEduSettings,
    test_short_settings: ShortSettings,
) -> Settings:
    """Полные настройки для тестов"""
    return Settings(
        app=test_app_settings,
        milvus=test_milvus_settings,
        opensearch=test_opensearch_settings,
        database=test_database_settings,
        redis=test_redis_settings,
        celery=test_celery_settings,
        vllm=test_vllm_settings,
        hybrid=test_hybrid_settings,
        llm_queue=test_llm_queue_settings,
        llm_global_sem=test_llm_global_sem_settings,
        reranker=test_reranker_settings,
        warmup=test_warmup_settings,
        switches=test_switches_settings,
        slowapi=test_slowapi_settings,
        search_metrics=test_search_metrics_settings,
        extract_edu=test_extract_edu_settings,
        postgres=test_database_settings,
        short_settings=test_short_settings,
    )


@pytest.fixture(scope="session")
def test_app_settings() -> AppSettings:
    """Настройки приложения для тестов"""
    return AppSettings(
        mode="test",
        host="0.0.0.0",
        port=8000,
        debug_host="0.0.0.0",
        debug_port=8001,
        workers_num=1,
        access_key="test_key",
        prefix="",
        use_cache=True,
        logs_path=None,
        logs_access_path=None,
        log_level="INFO",
        normalize_query=False,
        collection_files_contr_dir="./tests/mocks/collections",
        data_unique_id="ext_id",
        recreate_data=True,
        generate_prelaunch_data=False,
    )


@pytest.fixture(scope="session")
def test_milvus_settings(
    milvus_container: MilvusContainer,
) -> MilvusSettings:
    """Настройки Milvus для тестов"""
    host = milvus_container.get_container_host_ip()
    port = milvus_container.get_exposed_port(19530)

    model_name = os.getenv("MILVUS_MODEL_NAME")
    if not model_name:
        pytest.fail("MILVUS_MODEL_NAME не установлен в .env")

    model_store_path = os.getenv("APP_MODELSTORE_HOST_PATH")
    if not model_store_path:
        pytest.fail("APP_MODELSTORE_HOST_PATH не установлен в .env")

    model_path = Path(model_store_path) / model_name
    if not model_path.exists():
        pytest.fail(f"Модель не найдена: {model_path}")

    return MilvusSettings(
        host=host,
        port=port,
        web_ui_port=9091,
        use_ssl=False,
        connection_timeout=30,
        query_timeout=60,
        collection_name="kb_default",
        model_name=str(model_path),
        vector_field="embedding",
        schema_path="app/settings/conf.json",
        id_field="ext_id",
        metric_type="IP",
        search_fields="question",
        output_fields="row_idx,source,ext_id,page_id,role,component,question,analysis,answer",
    )


@pytest.fixture(scope="session")
def test_opensearch_settings(
    opensearch_container: OpenSearchContainer,
) -> OpenSearchSettings:
    """Настройки OpenSearch для тестов"""
    host = opensearch_container.get_container_host_ip()
    port = opensearch_container.get_exposed_port(9200)

    return OpenSearchSettings(
        host=host,
        port=port,
        index_name="aisearch-qa",
        use_ssl=False,
        verify_certs=False,
        user=None,
        password=None,
        search_fields="question,analysis,answer,source,role",
        output_fields=(
            "row_idx,source,ext_id,page_id,role,component,product,actual,second_line,"
            "question,question_md,analysis,analysis_md,answer,answer_md,for_user,jira,modified_at"
        ),
        operator="or",
        min_should_match=1,
        fuzziness=0,
        use_rescore=False,
        index_answer=True,
        schema_path="app/settings/os_index.json",
        bulk_chunk_size=1000,
    )


@pytest.fixture(scope="session")
def test_database_settings() -> PostgresSettings:
    """Настройки базы данных для тестов"""
    return PostgresSettings(
        engine="postgresql",
        host="localhost",
        port=5432,
        user="test",
        password="test",
        db="test",
        pool_size=5,
        use_async=True,
        echo=False,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        slave_hosts="",
    )


@pytest.fixture(scope="session")
def test_redis_settings() -> RedisSettings:
    """Настройки Redis для тестов"""
    return RedisSettings(
        hostname="localhost",
        port=6379,
        database=0,
    )


@pytest.fixture(scope="session")
def test_celery_settings() -> CelerySettings:
    """Настройки Celery для тестов"""
    return CelerySettings(
        logs_path=None,
        log_level="INFO",
        logs_queue_path=None,
        log_queue_level="INFO",
        workers_num=1,
    )


@pytest.fixture(scope="session")
def test_vllm_settings() -> VLLMSettings:
    """Настройки VLLM для тестов"""
    return VLLMSettings(
        base_url="http://localhost",
        port=8000,
        api_key=None,
        model=None,
        model_name="test-model",
        max_input_tokens=4096,
        max_output_tokens=512,
        temperature=0.7,
        top_p=0.9,
        request_timeout=60,
        stream=False,
    )


@pytest.fixture(scope="session")
def test_hybrid_settings() -> HybridSearchSettings:
    """Настройки гибридного поиска для тестов"""
    return HybridSearchSettings(
        dense_top_k=20,
        lex_top_k=50,
        top_k=5,
        w_ce=0.6,
        w_dense=0.25,
        w_lex=0.15,
        dense_abs_min=0.25,
        dense_rel_min=0.6,
        lex_rel_min=0.5,
        precut_min_keep=3,
        cache_ttl=3600,
        version="v1",
        collection_name="kb_default",
        merge_by_field="ext_id",
        merge_fields="row_idx,source,ext_id,page_id,role,component,question,analysis,answer",
        enable_intermediate_results=False,
        intermediate_results_top_k=10,
    )


@pytest.fixture(scope="session")
def test_llm_queue_settings() -> LLMQueueSettings:
    """Настройки очереди LLM для тестов"""
    return LLMQueueSettings(
        queue_list_key="llm:queue:list",
        ticket_hash_prefix="llm:ticket:",
        max_size=100,
        ticket_ttl=3600,
        drain_interval_sec=1,
        processing_list_key=None,
    )


@pytest.fixture(scope="session")
def test_llm_global_sem_settings() -> LLMGlobalSemaphoreSettings:
    """Настройки глобального семафора LLM для тестов"""
    return LLMGlobalSemaphoreSettings(
        key="llm:{global}:sem",
        limit=2,
        ttl_ms=120000,
        wait_timeout_ms=1,
        heartbeat_ms=30000,
    )


@pytest.fixture(scope="session")
def test_reranker_settings() -> RerankerSettings:
    """Настройки реранкера для тестов"""
    return RerankerSettings(
        model_name="test-reranker",
        device="cuda",
        batch_size=128,
        max_length=192,
        score_mode="sigmoid",
        softmax_temp=0.7,
        pairs_fields="question,answer",
        dtype="fp16",
    )


@pytest.fixture(scope="session")
def test_warmup_settings() -> WarmupSettings:
    """Настройки прогрева для тестов"""
    return WarmupSettings(
        enabled=False,
        ce_pairs="",
        embed_texts="",
    )


@pytest.fixture(scope="session")
def test_switches_settings() -> SearchSwitches:
    """Настройки переключателей поиска для тестов"""
    return SearchSwitches(
        use_opensearch=True,
        use_reranker=True,
        use_hybrid=True,
    )


@pytest.fixture(scope="session")
def test_slowapi_settings() -> SlowAPISettings:
    """Настройки лимитов для тестов"""
    return SlowAPISettings(
        search="100/minute",
        generate="10/minute",
    )


@pytest.fixture(scope="session")
def test_search_metrics_settings() -> SearchMetricsEnabled:
    """Настройки метрик поиска для тестов"""
    return SearchMetricsEnabled(
        log_metrics_enabled=False,
        response_metrics_enabled=False,
    )


@pytest.fixture(scope="session")
def test_extract_edu_settings() -> ExtractEduSettings:
    """Настройки извлечения EDU для тестов"""
    return ExtractEduSettings(
        edu_emias_url="http://test.edu",
        base_harvester_api_url="http://test.harvester",
        edu_emias_token="test-token",
        edu_emias_attachments_page_id="123",
        edu_timeout=30,
        knowledge_base_file_name="test_kb.xlsx",
        vio_base_file_name="test_vio.xlsx",
        cron_update_times="09:00",
        vio_harvester_suffix="/knowledge-base/collect-all",
        kb_harvester_suffix="/vio/runtime_harvest",
        logs_path=None,
        log_level="INFO",
    )


@pytest.fixture(scope="session")
def test_short_settings() -> ShortSettings:
    """Настройки коротких запросов для тестов"""
    return ShortSettings(
        mode=True,
        mode_limit=10,
        use_opensearch=True,
        use_reranker=True,
        use_hybrid=True,
        dense_top_k=20,
        lex_top_k=50,
        top_k=5,
        w_ce=0.6,
        w_dense=0.25,
        w_lex=0.15,
    )
