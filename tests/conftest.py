import asyncio
import logging
import typing as tp
from pathlib import Path

import pandas as pd
import pytest
from pymilvus import MilvusException

from app.infrastructure.adapters.open_search import OpenSearchAdapter
from app.infrastructure.storages.milvus import MilvusDatabase
from app.infrastructure.utils.universal import get_system_root, settings_to_env_vars
from app.services.updater import UpdaterService
from app.settings.config import Settings
from tests.mocks.edu import MockEduAdapter
from tests.utils import cleanup_milvus, cleanup_opensearch

pytest_plugins = ["tests.fixtures.containers", "tests.fixtures.settings"]


@pytest.fixture(scope="session", autouse=True)
def suppress_unwanted_logs():
    """Подавляем ненужные логи для тестовой сессии"""
    logging.getLogger("opensearch").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)


@pytest.fixture(scope="session")
def event_loop() -> tp.Generator[asyncio.AbstractEventLoop, None, None]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def milvus_database(
    test_milvus_settings,
) -> tp.AsyncGenerator[MilvusDatabase, None]:
    """MilvusDatabase для всей сессии тестов"""
    from app.common.logger import AISearchLogger, LoggerType

    database = MilvusDatabase(
        test_milvus_settings, AISearchLogger(logger_type=LoggerType.TEST)
    )

    try:
        await database._has_collection("test_connection")
    except MilvusException as e:
        pytest.fail(f"Не удалось подключиться к Milvus: {e}")

    yield database
    await database.close()


@pytest.fixture(scope="session")
async def opensearch_adapter(
    test_settings: Settings,
) -> tp.AsyncGenerator[OpenSearchAdapter, None]:
    """OpenSearchAdapter для всей сессии тестов"""
    from app.common.logger import AISearchLogger, LoggerType

    adapter = OpenSearchAdapter(
        test_settings, AISearchLogger(logger_type=LoggerType.TEST)
    )

    try:
        health = await adapter.client.cluster.health(
            wait_for_status="yellow", timeout=30
        )
        if health["status"] not in ("yellow", "green"):
            pytest.fail(f"OpenSearch не готов: статус {health['status']}")
    except Exception as e:
        pytest.fail(f"Не удалось подключиться к OpenSearch: {e}")

    yield adapter
    await adapter.close()


@pytest.fixture()
async def clean_databases(
    milvus_database: MilvusDatabase,
    opensearch_adapter: OpenSearchAdapter,
    test_milvus_settings,
    test_opensearch_settings,
) -> None:
    """Очищает тестовые данные перед тестом"""
    await cleanup_milvus(milvus_database, test_milvus_settings.collection_name)
    await cleanup_opensearch(opensearch_adapter, test_opensearch_settings.index_name)


@pytest.fixture()
def pre_launch_env_vars(
    test_settings: Settings,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, str]:
    """Переменные окружения для запуска pre_launch.py"""
    tmp_path = tmp_path_factory.mktemp("pre_launch_test")

    log_path = tmp_path / "pre_launch.log"
    celery_log_path = tmp_path / "celery.log"
    nltk_data_path = tmp_path / "nltk_data"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    celery_log_path.parent.mkdir(parents=True, exist_ok=True)
    nltk_data_path.mkdir(parents=True, exist_ok=True)

    env_vars = settings_to_env_vars(test_settings)

    env_vars["SYSTEMROOT"] = get_system_root()
    env_vars["PYTHONIOENCODING"] = "utf-8"
    env_vars["PYTHONUTF8"] = "1"

    env_vars.update(
        {
            "APP_RECREATE_DATA": "true",
            "LOG_PATH": str(log_path),
            "CELERY_LOGS_PATH": str(celery_log_path),
            "NLTK_DATA": str(nltk_data_path),
        }
    )

    return env_vars


@pytest.fixture()
def prelaunch_updater_correct_result_df() -> pd.DataFrame:
    """Ожидаемый результат после обработки файлов"""
    expected_file = (
        Path(__file__).parent / "mocks" / "prelaunch_updater_correct_result.xlsx"
    )
    if not expected_file.exists():
        pytest.skip(f"Файл с ожидаемым результатом не найден: {expected_file}")
    return pd.read_excel(expected_file)


@pytest.fixture()
def mock_edu_adapter() -> MockEduAdapter:
    return MockEduAdapter()


@pytest.fixture()
def updater_service(
    test_settings: Settings,
    mock_edu_adapter: MockEduAdapter,
    milvus_database: MilvusDatabase,
    opensearch_adapter: OpenSearchAdapter,
) -> UpdaterService:
    """Создает экземпляр UpdaterService для тестов"""
    from app.common.logger import AISearchLogger, LoggerType
    from app.services.updater import UpdaterService

    return UpdaterService(
        settings=test_settings,
        logger=AISearchLogger(logger_type=LoggerType.TEST),
        edu=mock_edu_adapter,
        milvus=milvus_database,
        os=opensearch_adapter,
    )
