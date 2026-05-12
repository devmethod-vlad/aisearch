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
  "metrics_enable": false,
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

## top_k
- default: берётся из настроек поиска (`HYBRID_TOP_K` или short-mode override), если поле отсутствует в запросе;
- если `top_k` передан явно — используется это значение;
- `top_k` должен быть положительным целым.

## search_use_cache
- default: `true`;
- `false` отключает чтение search-cache;
- `true` разрешает чтение search-cache независимо от `show_intermediate_results`;
- запись нового результата в Redis выполняется всегда.

## show_intermediate_results
- default: `false`;
- `true` добавляет `intermediate_results` (`dense` / `lex` / `fusion` / `ce`);
- дополнительно возвращается `intermediate_results.fusion`, отсортированный по `score_fusion`;
- не влияет на решение читать cache: за cache read отвечает только `search_use_cache`;
- при cache hit `intermediate_results` возвращается только если поле есть в cache payload;
- если при cache hit поля `intermediate_results` нет, pipeline повторно не запускается.

## Формат search-cache payload
Новая запись search-cache всегда хранится в объектном формате:
- обязательно: `results`;
- опционально: `intermediate_results` (только если fresh execution шёл с `show_intermediate_results=true`);
- поле `schema_version` не используется и не добавляется.

Примеры:
```json
{"results":[...]}
```
```json
{"results":[...],"intermediate_results":{"dense":[...],"lex":[...],"fusion":[...],"ce":[...]}}
```

## metrics_enable
- default: `false`;
- `true` добавляет блок `metrics` в итоговый payload;
- `false` не включает `metrics` в ответ.

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
  "metrics_enable": false,
  "presearch": null
}
```

## Cache key
Cache key учитывает:
- query/effective top_k;
- filters;
- состояние presearch (вкл/выкл), `presearch.field` и `raw_query`.
- режим fusion (`HYBRID_FUSION_MODE`) и `HYBRID_RRF_K`, чтобы не смешивать кеш разных режимов ранжирования.

## Миграция с env
Переменные `APP_USE_CACHE`, `HYBRID_ENABLE_INTERMEDIATE_RESULTS`, `HYBRID_PRESEARCH_ENABLED`, `HYBRID_PRESEARCH_FIELD` больше не управляют runtime-поиском.

## Примеры
- обычный поиск: `{"query":"..."}`
- явный top_k: `{"query":"...","top_k":5}`
- fresh search: `{"query":"...","search_use_cache":false}`
- intermediate: `{"query":"...","show_intermediate_results":true}`
- response metrics: `{"query":"...","metrics_enable":true}`
- presearch: `{"query":"12345","presearch":{"field":"ext_id"}}`
