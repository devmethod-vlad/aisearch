# Search-cache data versioning

## Причины изменений

- Search-cache хранится во внешнем Redis и переживает рестарты приложения.
- При `APP_RECREATE_DATA=true` Milvus/OpenSearch могут быть пересозданы, но старые cache-ключи продолжают жить до TTL.
- Полная очистка Redis недопустима, потому что в том же Redis хранятся очереди, статусы задач, семафоры и другие служебные ключи.

## Цели

- Безопасно инвалидировать только search-cache после смены дата-среза.
- Не затрагивать служебные Redis-ключи.
- Сохранить существующую бизнес-логику поиска и текущий TTL для кеша ответов.

## Выбранная архитектура

- Введён отдельный meta-ключ версии данных:
  - `hyb:meta:data_version:{collection_name}:{index_name}`.
- В search-cache key добавлена обязательная часть `data_version`:
  - `hyb:cache:{collection_name}:{index_name}:{data_version}:{hybrid_version}:{query_hash}:{top_k}:{presearch_key_part}:{filters_key_part}`.
- Версия данных обновляется (bump) только когда реально готов новый дата-срез.
- Meta-ключ `data_version` хранится **без TTL**; TTL применяется только к search-cache ключам.

## Основные шаги реализации

1. Добавлен helper-модуль `app/infrastructure/utils/search_cache_version.py`:
   - сборка meta/cache ключей;
   - атомарное `get_or_create` initial-версии через `SET NX` + повторное чтение;
   - `bump` версии в формате `UTC timestamp + uuid4 hex`.
2. В `HybridSearchOrchestrator.documents_search`:
   - удалена ручная сборка cache key;
   - перед чтением (если разрешено request-параметром `search_use_cache`) и перед записью кеша извлекается текущая `data_version`;
   - ключ строится через единый helper.
3. В `pre_launch.py`:
   - после успешного пересоздания Milvus и/или OpenSearch выполняется bump версии;
   - в `--export-only` bump не выполняется;
   - bump не выполняется, если пересоздание не требовалось;
   - Redis-клиент всегда корректно закрывается.
4. В `UpdaterService`:
   - добавлена зависимость `KeyValueStorageProtocol`;
   - `_update_collection_from_df` возвращает `bool` о реальных успешных изменениях;
   - `update_kb_base`/`update_vio_base` bump-ают версию только при `True`;
   - `update_all` выполняет один bump при изменениях хотя бы в одной части и делает это до опционального Excel-экспорта.

## Точки интеграции

- Runtime search: `app/services/hybrid_search_orchestrator.py`.
- Pre-launch подготовка данных: `pre_launch.py`.
- Runtime updater: `app/services/updater.py`.
- Общая утилита versioning: `app/infrastructure/utils/search_cache_version.py`.

## Что не нужно делать

- Не очищать весь Redis.
- Не удалять ключи очередей, статусов задач и семафоров.
- Не использовать `HYBRID_VERSION` как замену `data_version`.
- Не генерировать новую `data_version` на каждый search-запрос.
- Не ставить TTL на meta-ключ текущей версии данных.
- Не дублировать формат cache key по нескольким местам.

## Критерии готовности

- Runtime-поиск использует cache key с `data_version`.
- После успешного pre-launch recreate выполняется bump `data_version`.
- После runtime update bump происходит только при реальных успешных изменениях данных.
- Без изменений данных bump не выполняется.
- Полная очистка Redis нигде не используется.
- Поведение поиска, token-фильтров, presearch и TTL кеша ответов не изменено.
