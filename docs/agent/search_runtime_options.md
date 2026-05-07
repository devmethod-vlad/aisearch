# Search runtime options

## Цель
Runtime-поведение гибридного поиска управляется параметрами запроса, а не env-переменными.

## API-контракт
```json
{
  "query": "...",
  "top_k": 5,
  "search_use_cache": true,
  "show_intermediate_results": false,
  "presearch": {
    "field": "ext_id"
  },
  "filters": {
    "array_filters": {},
    "exact_filters": {}
  }
}
```
Все runtime-параметры необязательные.

## search_use_cache
- default: `true`;
- `false` отключает только чтение старого кеша;
- запись нового результата в Redis выполняется всегда.

## show_intermediate_results
- default: `false`;
- `true` добавляет `intermediate_results` (`dense` / `lex` / `ce`);
- дополнительно возвращается `intermediate_results.fusion`, отсортированный по `score_fusion`;
- при `true` чтение кеша пропускается, потому что intermediate results требуют свежего выполнения pipeline;
- финальные `results` всё равно записываются в кеш.

## presearch
- default: disabled;
- включается только при наличии объекта `presearch`;
- `presearch.field` обязателен;
- presearch не использует `array_filters`/`exact_filters`;
- presearch влияет на cache key.

## Redis pack
```json
{
  "type": "search",
  "query": "...",
  "top_k": 5,
  "array_filters": {},
  "exact_filters": {},
  "search_use_cache": true,
  "show_intermediate_results": false,
  "presearch": null
}
```

## Cache key
Cache key учитывает:
- query/top_k;
- filters;
- состояние presearch (вкл/выкл), `presearch.field` и `raw_query`.
- режим fusion (`HYBRID_FUSION_MODE`) и `HYBRID_RRF_K`, чтобы не смешивать кеш разных режимов ранжирования.

## Миграция с env
Переменные `APP_USE_CACHE`, `HYBRID_ENABLE_INTERMEDIATE_RESULTS`, `HYBRID_PRESEARCH_ENABLED`, `HYBRID_PRESEARCH_FIELD` больше не управляют runtime-поиском.

## Примеры
- обычный поиск: `{"query":"..."}`
- fresh search: `{"query":"...","search_use_cache":false}`
- intermediate: `{"query":"...","show_intermediate_results":true}`
- presearch: `{"query":"12345","presearch":{"field":"ext_id"}}`
