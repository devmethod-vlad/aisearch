# Exact filtering pipeline

## 1. Цель изменений
Добавлена отдельная exact-фильтрация по single-value metadata полям `source`, `actual`, `second_line`.

## 2. Отличие exact от token
- Token-фильтры: мультизначные поля, split по разделителю, OR внутри поля.
- Exact-фильтры: одно строковое значение, без split, только точное сравнение нормализованного значения.

## 3. Почему не raw-поля
Raw-поля нужны для отображения, отладки, экспорта и update-логики. Для фильтрации используются отдельные служебные поля `*_filter`.

## 4. Env-настройки
- `EXACT_FILTERS_ENV_SEPARATOR` — разделитель списка полей в env (по умолчанию `,`).
- `EXACT_FILTERS_RAW_FIELDS` — список raw-полей (`source,actual,second_line`).
- `EXACT_FILTERS_FIELD_SUFFIX` — суффикс служебного поля (`_filter`).

## 5. Формирование полей
- `source -> source_filter`
- `actual -> actual_filter`
- `second_line -> second_line_filter`

## 6. Нормализация
- `strip()`
- `casefold()`
- без split

## 7. Ingestion
- `prepare_dataframe`: добавляет `*_filter` после token enrichment.
- `pre_launch`: передаёт `ExactFilterConfig` в `prepare_dataframe`.
- `UpdaterService`: использует тот же `ExactFilterConfig` для всех update-путей.

## 8. OpenSearch
Служебные поля хранятся как `keyword`; runtime применяет `term`-фильтры в `bool.filter`.

## 9. Milvus
Служебные поля хранятся как `VARCHAR` + `INVERTED` индексы; runtime выражения формируются через `field == "value"`.

## 10. Runtime pipeline
API -> router читает `filters.exact_filters` (nested-контракт) -> `enqueue_search` пишет словарь в Redis pack -> `HybridSearchOrchestrator` нормализует и применяет в dense/lex поиске.

## 11. Семантика
- exact между собой: AND
- token: OR внутри поля и AND между полями
- token + exact: AND

## 12. Cache key
Cache key включает оба типа фильтров: `tokens:<...>;exact:<...>`.

## 13. Presearch
Presearch остаётся без exact-фильтрации (unfiltered exact-match этап).

## 14. Reindex/recreate
После изменения схем требуется пересоздание OpenSearch index и Milvus collection / reindex.

## 15. Чеклист добавления нового exact-поля
1. Добавить raw-поле в `EXACT_FILTERS_RAW_FIELDS`.
2. Обновить схемы OpenSearch/Milvus.
3. Прогнать ingestion и тесты.

## 16. Покрытие тестами
- `tests/unit/test_exact_filters.py`
- `tests/unit/test_prepare_dataframe_tokens.py`
- `tests/unit/test_hybrid_search_service_filters.py`
- `tests/unit/test_orchestrator_filter_queries.py`
- `tests/unit/test_orchestrator_presearch_pipeline.py`


## Примеры API
Только exact_filters:
```json
{
  "query": "как оформить назначение",
  "top_k": 5,
  "filters": {
    "exact_filters": {
      "actual": "Да",
      "second_line": "Да"
    }
  }
}
```

Совмещённо с array_filters — см. `docs/agent/search_filters_api_contract.md`.
