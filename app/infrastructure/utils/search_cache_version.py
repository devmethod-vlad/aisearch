import logging
from datetime import UTC, datetime
from uuid import uuid4

from app.common.storages.interfaces import KeyValueStorageProtocol

logger = logging.getLogger(__name__)


def build_search_data_version_key(collection_name: str, index_name: str) -> str:
    """Собирает Redis meta-ключ текущей версии поискового дата-среза.

    Ключ хранит "текущую" версию данных (Milvus/OpenSearch) и используется:
    - в runtime-поиске (`HybridSearchOrchestrator.documents_search`) для
      построения namespace search-cache;
    - в pre-launch/обновлениях (`pre_launch.py`, `UpdaterService`) для
      безопасной инвалидации только search-cache без удаления служебных ключей
      Redis (очереди, статусы задач, семафоры и т.д.).

    :param collection_name: Имя коллекции Milvus, используемое search pipeline.
    :param index_name: Имя индекса OpenSearch, соответствующего тому же срезу данных.
    :return: Строка ключа в формате
        ``hyb:meta:data_version:{collection_name}:{index_name}``.
    """
    return f"hyb:meta:data_version:{collection_name}:{index_name}"


def build_search_cache_key(
    *,
    collection_name: str,
    index_name: str,
    data_version: str,
    hybrid_version: str,
    query_hash: str,
    top_k: int,
    presearch_key_part: str,
    filters_key_part: str,
) -> str:
    """Собирает итоговый Redis key для search-cache.

    Ключ детерминированно учитывает:
    - выбранный дата-срез (`data_version`);
    - runtime-версию гибридного пайплайна (`hybrid_version`);
    - параметры конкретного поискового запроса.

    Функция централизует единый формат ключа, чтобы исключить дублирование
    шаблона по коду и избежать рассинхронизации.

    :param collection_name: Имя коллекции Milvus.
    :param index_name: Имя индекса OpenSearch.
    :param data_version: Текущая версия данных (из meta-ключа).
    :param hybrid_version: Логическая версия алгоритма поиска (`HYBRID_VERSION`).
    :param query_hash: Хэш запроса (или нормализованный query при выключенной нормализации).
    :param top_k: Запрошенный размер выдачи.
    :param presearch_key_part: Детерминированная часть ключа presearch-настроек.
    :param filters_key_part: Детерминированная часть токен-фильтров.
    :return: Строка ключа в формате
        ``hyb:cache:{collection_name}:{index_name}:{data_version}:{hybrid_version}:``
        ``{query_hash}:{top_k}:{presearch_key_part}:{filters_key_part}``.
    """
    return (
        f"hyb:cache:{collection_name}:{index_name}:{data_version}:{hybrid_version}:"
        f"{query_hash}:{top_k}:{presearch_key_part}:{filters_key_part}"
    )


def _generate_data_version() -> str:
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{ts}-{uuid4().hex}"


async def get_or_create_search_data_version(
    redis: KeyValueStorageProtocol,
    *,
    collection_name: str,
    index_name: str,
) -> str:
    """Возвращает текущую версию данных, создавая initial-значение при отсутствии.

    Алгоритм атомарно инициализирует версию через ``SET ... NX`` (`not_exist=True`),
    после чего перечитывает ключ. Это защищает от гонки на параллельном старте
    нескольких worker-процессов: initial-версия будет одна, а не по версии на каждый
    процесс.

    :param redis: Клиент key-value хранилища (Redis-совместимый).
    :param collection_name: Имя коллекции Milvus.
    :param index_name: Имя индекса OpenSearch.
    :return: Актуальная строка версии дата-среза.

    Важно: meta-ключ версии не должен иметь TTL, потому что это "указатель" на
    текущий валидный дата-срез. Если он протухнет, runtime может неявно создать
    новую версию и потерять связность cache-space без фактического изменения данных.
    """
    version_key = build_search_data_version_key(collection_name, index_name)
    existing = await redis.get(version_key)
    if existing:
        return str(existing)

    initial_version = _generate_data_version()
    await redis.set(version_key, initial_version, not_exist=True)
    current_version = await redis.get(version_key)
    if current_version:
        if str(current_version) == initial_version:
            logger.info(
                "Initialized search data version for %s/%s: %s",
                collection_name,
                index_name,
                current_version,
            )
        return str(current_version)

    # Крайне маловероятный fallback: если Redis вернул пусто после set/get.
    logger.warning(
        "Search data version key %s was empty after NX initialization; using local fallback",
        version_key,
    )
    return initial_version


async def bump_search_data_version(
    redis: KeyValueStorageProtocol,
    *,
    collection_name: str,
    index_name: str,
    reason: str | None = None,
) -> str:
    """Принудительно обновляет версию данных для изоляции нового data-snapshot.

    Используется после успешной подготовки/перезагрузки Milvus/OpenSearch или после
    runtime-обновлений, которые реально изменили данные. Старая cache-область не
    удаляется физически, но становится недостижимой для новых запросов за счёт
    смены ``data_version``.

    :param redis: Клиент key-value хранилища.
    :param collection_name: Имя коллекции Milvus.
    :param index_name: Имя индекса OpenSearch.
    :param reason: Опциональная причина bump (для логирования и трассировки).
    :return: Новое значение версии дата-среза.

    Важно: ключ версии хранится без TTL. Он представляет "текущий срез" данных,
    а не кешируемый ответ. TTL применим только к ключам search-cache.
    """
    version_key = build_search_data_version_key(collection_name, index_name)
    new_version = _generate_data_version()
    await redis.set(version_key, new_version)
    logger.info(
        "Bumped search data version for %s/%s: %s (reason=%s)",
        collection_name,
        index_name,
        new_version,
        reason or "n/a",
    )
    return new_version
