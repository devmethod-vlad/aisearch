# Фильтрация поиска по мультизначным metadata-полям

Дата актуализации: 2026-04-24  
Контекст: `devmethod-vlad/aisearch`, реализация фильтрации по полям `role` и `product`.

## 1. Цель реализации

Во внешних источниках часть metadata-полей приходит в виде строки, где несколько значений перечислены через разделитель.

Пример:

```text
role = "Врач;Медсестра"
product = "ЭМИАС;ЛИС"
```

Такие поля неудобно и небезопасно фильтровать как обычную строку:

- поиск подстрокой может давать ложные совпадения;
- разные регистры и пробелы мешают точному совпадению;
- одинаковую логику сложно поддерживать одновременно в Milvus и OpenSearch;
- при добавлении новых мультизначных полей легко получить расхождение между ingestion и runtime-поиском.

Поэтому была введена отдельная token-based схема:

- исходные raw-поля `role` и `product` сохраняются как есть;
- рядом с ними создаются служебные массивы токенов:
  - `role_tokens`;
  - `product_tokens`;
- поиск фильтруется не по raw-строке, а по нормализованным токенам;
- предварительный exact-match поиск `presearch` намеренно не ограничивается этими фильтрами.

Итоговая цель: сделать точную, расширяемую и одинаковую фильтрацию в Milvus и OpenSearch для мультизначных metadata-полей.

## 2. Общая модель данных

### Raw-поля

Raw-поля — это исходные поля из внешнего источника:

```text
role
product
```

Они остаются в metadata и используются для отображения в ответе API.

Пример raw-значения:

```text
"Врач;Медсестра"
```

### Token-поля

Token-поля — это служебные поля для фильтрации:

```text
role_tokens
product_tokens
```

Пример преобразования:

```text
role = "Врач;Медсестра"
role_tokens = ["врач", "медсестра"]

product = "ЭМИАС;ЛИС"
product_tokens = ["эмиас", "лис"]
```

Token-поля не должны использоваться как пользовательские поля ответа. Их задача — быть внутренним индексируемым представлением для фильтрации.

## 3. Настройки token-фильтров

Конфигурация вынесена в env-настройки:

```env
TOKEN_FILTERS_ENV_SEPARATOR=,
TOKEN_FILTERS_RAW_FIELDS=role,product
TOKEN_FILTERS_TOKEN_SUFFIX=_tokens
TOKEN_FILTERS_RAW_SEPARATOR=;
```

Назначение:

| Переменная | Значение | Назначение |
| --- | --- | --- |
| `TOKEN_FILTERS_RAW_FIELDS` | `role,product` | Список raw-полей, для которых нужно создавать token-поля |
| `TOKEN_FILTERS_TOKEN_SUFFIX` | `_tokens` | Суффикс служебного token-поля |
| `TOKEN_FILTERS_RAW_SEPARATOR` | `;` | Разделитель значений внутри raw-строки источника |
| `TOKEN_FILTERS_ENV_SEPARATOR` | `,` | Разделитель списка полей в env-переменной |

Имена token-полей формируются детерминированно:

```text
{raw_field}{token_suffix}
```

Например:

```text
role + _tokens = role_tokens
product + _tokens = product_tokens
```

## 4. Единая утилита token-фильтрации

Центральная логика вынесена в модуль:

```text
app/infrastructure/utils/token_filters.py
```

Основные сущности:

### `MultiValueTokenConfig`

Описывает правила token-фильтрации:

```python
MultiValueTokenConfig(
    raw_fields=("role", "product"),
    token_suffix="_tokens",
    raw_separator=";",
)
```

Этот конфиг используется в двух местах:

1. ingestion-пайплайн;
2. runtime-поиск.

Это важно: загрузка данных и поиск должны одинаково понимать, что такое `role_tokens` и `product_tokens`.

### Нормализация одного токена

Базовая нормализация:

```text
str(value).strip().casefold()
```

То есть:

```text
" Врач " -> "врач"
"МЕДСЕСТРА" -> "медсестра"
```

### Токенизация raw metadata-поля

Для значения из внешнего источника применяется split по `TOKEN_FILTERS_RAW_SEPARATOR`.

Алгоритм:

1. привести значение к строке;
2. разбить по `;`;
3. обрезать пробелы;
4. привести к `casefold`;
5. удалить пустые значения;
6. удалить дубли с сохранением порядка.

Пример:

```text
" Врач ; Медсестра ; ВРАЧ "
```

превращается в:

```python
["врач", "медсестра"]
```

Важное правило: частичное совпадение не создаётся.

```text
"Врач на дому;Медсестра"
```

превращается в:

```python
["врач на дому", "медсестра"]
```

Токен `"врач"` отдельно не появляется.

## 5. Отличие ingestion-токенизации от runtime-фильтра

Есть два разных входных формата:

### Ingestion

Во время загрузки данных поле приходит как одна строка:

```python
{"role": "Врач;Медсестра"}
```

Здесь значение нужно split-ить по `;`.

### Runtime API

В API фильтр приходит уже как список выбранных значений:

```json
{
  "query": "как оформить назначение",
  "role": ["Врач", "Медсестра"],
  "product": ["ЭМИАС"]
}
```

Здесь каждый элемент списка уже считается отдельным значением. Поэтому runtime-фильтр не split-ится по `;`.

Это принципиально: значение `"ЭМИАС;ЛИС"`, переданное в API как один элемент списка, трактуется как один токен `"эмиас;лис"`, а не как два токена.

## 6. Семантика фильтрации

Фильтрация строится так:

- внутри одного поля — `OR`;
- между разными полями — `AND`.

Пример запроса:

```json
{
  "role": ["Врач", "Медсестра"],
  "product": ["ЭМИАС", "ЛИС"]
}
```

Логика:

```text
(role_tokens содержит "врач" OR role_tokens содержит "медсестра")
AND
(product_tokens содержит "эмиас" OR product_tokens содержит "лис")
```

То есть документ должен подходить хотя бы по одному значению каждого указанного поля.

## 7. Первичная загрузка данных: `pre_launch.py`

Файл:

```text
pre_launch.py
```

Общий пайплайн первичной загрузки:

1. загрузить файлы из директории коллекции;
2. переименовать поля по `field_mapping.json`;
3. провалидировать наличие обязательных полей;
4. объединить источники;
5. выполнить дедупликацию;
6. подготовить финальный DataFrame через `prepare_dataframe`;
7. загрузить metadata в Milvus и OpenSearch.

Важное изменение: перед вызовом `prepare_dataframe` создаётся `MultiValueTokenConfig` из settings:

```python
token_filter_config = MultiValueTokenConfig(
    raw_fields=settings.token_filters.raw_fields,
    token_suffix=settings.token_filters.token_suffix,
    raw_separator=settings.token_filters.raw_separator,
)
```

Затем он передаётся в `prepare_dataframe`:

```python
documents, metadata, df_final = prepare_dataframe(
    combined_df,
    id_column=settings.app.data_unique_id,
    token_config=token_filter_config,
)
```

Это гарантирует, что при первичной загрузке записи получают `role_tokens` и `product_tokens` до записи в Milvus и OpenSearch.

## 8. Единая подготовка DataFrame: `prepare_dataframe`

Файл:

```text
app/infrastructure/utils/prepare_dataframe.py
```

Функция:

```python
prepare_dataframe(
    df: pd.DataFrame,
    id_column: str,
    token_config: MultiValueTokenConfig | None = None,
)
```

Она используется в двух ingestion-путях:

1. `pre_launch.py`;
2. `UpdaterService`.

Основные действия:

1. привести `page_id` к строковому виду;
2. привести DataFrame к string-значениям;
3. обрезать `id_column` и `source`;
4. отфильтровать пустые вопросы;
5. разделить записи по источникам `ВиО` и `ТП`;
6. применить source-specific фильтры:
   - `actual` не содержит `нет`;
   - для `ВиО` поле `space` не пустое и не равно `не распределено`;
7. проставить `row_idx`;
8. если передан `token_config`, добавить token-поля;
9. вернуть:
   - список документов для эмбеддинга;
   - metadata;
   - подготовленный DataFrame.

Token-поля добавляются централизованно:

```python
enriched_records = enrich_records_with_token_fields(
    df_final.to_dict(orient="records"),
    config=token_config,
)
df_final = pd.DataFrame(enriched_records)
```

Это ключевая точка: правила enrichment находятся не в `pre_launch.py` и не в updater-сервисе отдельно, а в одном общем месте.

## 9. Обновление данных: `UpdaterService`

Файл:

```text
app/services/updater.py
```

Сервис updater тоже создаёт `MultiValueTokenConfig` из settings:

```python
self.token_filter_config = MultiValueTokenConfig(
    raw_fields=settings.token_filters.raw_fields,
    token_suffix=settings.token_filters.token_suffix,
    raw_separator=settings.token_filters.raw_separator,
)
```

Этот же конфиг используется во всех основных update-сценариях:

```text
update_vio_base()
update_kb_base()
update_all()
```

### `update_vio_base`

Пайплайн:

1. скачать файл ВиО;
2. переименовать поля;
3. провалидировать DataFrame;
4. вызвать `prepare_dataframe(..., token_config=self.token_filter_config)`;
5. передать подготовленный DataFrame в `_update_collection_from_df`.

### `update_kb_base`

То же самое для KB/ТП:

1. скачать файл KB;
2. переименовать поля;
3. провалидировать DataFrame;
4. вызвать `prepare_dataframe(..., token_config=self.token_filter_config)`;
5. передать подготовленный DataFrame в `_update_collection_from_df`.

### `update_all`

Пайплайн:

1. скачать оба источника;
2. переименовать поля;
3. провалидировать оба DataFrame;
4. объединить источники;
5. выполнить дедупликацию;
6. разделить на `ТП` и `ВиО`;
7. для каждого источника вызвать `prepare_dataframe(..., token_config=self.token_filter_config)`;
8. обновить соответствующие записи в Milvus и OpenSearch.

### Upsert в базы

В `_update_collection_from_df` используется уже подготовленный DataFrame с token-полями.

Для записей, которые нужно создать или обновить:

```python
metadata = df_to_upsert.to_dict(orient="records")
```

Далее metadata передаётся:

```python
await self.milvus.upsert_vectors(...)
await self.os.upsert(...)
```

Значит, `role_tokens` и `product_tokens` попадают и в Milvus, и в OpenSearch при инкрементальных обновлениях.

## 10. Хранение в OpenSearch

Файл схемы:

```text
app/settings/os_index.json
```

В mapping добавлены служебные token-поля:

```json
"role_tokens": {
  "type": "keyword"
},
"product_tokens": {
  "type": "keyword"
}
```

OpenSearch умеет индексировать массивы строк в `keyword`-поле. Поэтому документ может выглядеть так:

```json
{
  "role": "Врач;Медсестра",
  "role_tokens": ["врач", "медсестра"]
}
```

Для корректной индексации важно не превратить список в строку вида:

```text
"['врач', 'медсестра']"
```

Поэтому в `OpenSearchAdapter._coerce_for_mapping` добавлена логика: если значение — `list`, а mapping поля — `keyword` или `text`, список сохраняется как массив строк.

## 11. Хранение в Milvus

Файл схемы:

```text
app/settings/conf.json
```

В Milvus добавлены поля:

```json
{
  "name": "role_tokens",
  "dtype": "ARRAY",
  "description": "Normalized role tokens for exact filtering",
  "element_type": "VARCHAR",
  "max_capacity": 32,
  "max_length": 255,
  "nullable": true
}
```

```json
{
  "name": "product_tokens",
  "dtype": "ARRAY",
  "description": "Normalized product tokens for exact filtering",
  "element_type": "VARCHAR",
  "max_capacity": 32,
  "max_length": 255,
  "nullable": true
}
```

Также была добавлена поддержка `ARRAY` в утилиту чтения Milvus-схемы:

```text
app/infrastructure/utils/milvus.py
```

Что поддерживается:

- `dtype = ARRAY`;
- `element_type`;
- `max_capacity`;
- `max_length`.

В `MilvusDatabase` добавлена логика приведения значений к типам схемы:

- `ARRAY`-поле принимает строку или последовательность;
- строка превращается в список из одного элемента;
- последовательность превращается в список;
- список ограничивается `max_capacity`;
- элементы `VARCHAR` приводятся к строке и обрезаются по `max_length`.

Эта логика используется и при insert, и при upsert.

## 12. API-контракт поиска

Файл:

```text
app/api/v1/dto/requests/search.py
```

В поисковый DTO добавлены поля:

```python
role: list[str] | None = None
product: list[str] | None = None
```

Пример запроса:

```json
{
  "query": "как создать назначение",
  "top_k": 5,
  "role": ["Врач"],
  "product": ["ЭМИАС"]
}
```

Router передаёт эти поля в сервис:

```python
service.enqueue_search(
    query=query,
    top_k=top_k,
    role=body.role,
    product=body.product,
)
```

## 13. Очередь поиска и Redis pack

Файл:

```text
app/services/hybrid_search_service.py
```

`HybridSearchService.enqueue_search` сохраняет фильтры в pack:

```python
pack = {
    "type": "search",
    "query": query,
    "top_k": top_k,
    "role": role,
    "product": product,
}
```

Этот pack кладётся в Redis и затем читается worker-ом в `HybridSearchOrchestrator.documents_search`.

## 14. Runtime-поиск в `HybridSearchOrchestrator`

Файл:

```text
app/services/hybrid_search_orchestrator.py
```

На старте orchestrator создаёт такой же `MultiValueTokenConfig`, как ingestion-пайплайны:

```python
self.token_filter_config = MultiValueTokenConfig(
    raw_fields=settings.token_filters.raw_fields,
    token_suffix=settings.token_filters.token_suffix,
    raw_separator=settings.token_filters.raw_separator,
)
```

В `documents_search` фильтры читаются из pack:

```python
raw_filters = {
    raw_field: pack.get(raw_field)
    for raw_field in self.token_filter_config.raw_fields
}
```

Затем они нормализуются:

```python
token_filters = normalize_request_token_filters(
    raw_filters,
    config=self.token_filter_config,
)
```

После этого строятся:

```python
filter_cache_key = token_filters.cache_key_part()
milvus_filter_expr = build_milvus_token_filter_expr(token_filters)
```

## 15. Cache key

Фильтры включены в cache key, чтобы разные фильтрованные запросы не смешивались.

Пример:

```text
role = ["Врач"]
product = ["ЭМИАС"]
```

даёт cache-фрагмент:

```text
product_tokens=эмиас|role_tokens=врач
```

Поля сортируются, чтобы ключ не зависел от порядка в словаре.

После доработки presearch итоговый cache key состоит из двух независимых частей:

- `presearch_key_part` — зависит от включённости presearch, поля presearch и текста запроса;
- `filters_key_part` — зависит от token-фильтров.

Это важно: presearch сам не фильтруется, но итоговый кеш всё равно должен различать разные фильтрованные запросы.

## 16. Фильтрация Milvus

Для Milvus строится expression.

Пример для одного значения:

```text
ARRAY_CONTAINS(role_tokens, "врач")
```

Пример для нескольких полей:

```text
ARRAY_CONTAINS(product_tokens, "эмиас") AND ARRAY_CONTAINS(role_tokens, "врач")
```

Пример с несколькими значениями внутри поля:

```text
(ARRAY_CONTAINS(product_tokens, "эмиас") OR ARRAY_CONTAINS(product_tokens, "лис"))
AND
(ARRAY_CONTAINS(role_tokens, "врач") OR ARRAY_CONTAINS(role_tokens, "медсестра"))
```

Этот expression передаётся в dense-поиск:

```python
await self.vector_db.search(
    collection_name=settings_local.collection_name,
    query_vector=query_vector,
    top_k=settings_local.dense_top_k,
    filter_expr=milvus_filter_expr,
)
```

В `MilvusDatabase.search` этот `filter_expr` передаётся в Milvus client как параметр `filter`.

## 17. Фильтрация OpenSearch

Для OpenSearch строятся `bool.filter` clauses.

Пример:

```json
[
  {
    "bool": {
      "should": [
        {
          "term": {
            "role_tokens": "врач"
          }
        }
      ],
      "minimum_should_match": 1
    }
  }
]
```

Для нескольких полей каждый field-group становится отдельным элементом `bool.filter`.

Это даёт `AND` между полями.

Внутри одного поля используется `should + minimum_should_match = 1`, что даёт `OR`.

Фильтры применяются в lexical-ветке:

```python
body = {
    "query": {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query,
                    "fields": os_adapter.config.search_fields,
                    "operator": os_adapter.config.operator,
                    "fuzziness": os_adapter.config.fuzziness,
                }
            },
            "filter": filter_clauses,
        }
    }
}
```

## 18. Presearch не фильтруется

Предварительный поиск `presearch` — это отдельный exact-match этап по `presearch_field`.

Он намеренно не использует `role`/`product` token-фильтры.

Причина: presearch должен найти точный документ по идентификатору или другому configured exact-match полю независимо от выбранных фильтров основного dense/lex поиска.

Актуальная логика:

- `documents_search` сначала может выполнить `_presearch_exact_match`;
- `_presearch_exact_match` не принимает `token_filters`;
- тела OpenSearch-запросов внутри presearch не содержат `filter`;
- если presearch-результат найден, он инжектится в итоговую выдачу сверху;
- основной dense/lex pipeline при этом остаётся фильтрованным.

## 19. Что нужно сделать при добавлении нового фильтруемого поля

Допустим, нужно добавить поле:

```text
department
```

Чеклист:

1. Убедиться, что raw-поле приходит из внешнего источника.
2. Добавить поле в field mapping, если оно ещё не маппится во внутреннее имя.
3. Добавить поле в env:

   ```env
   TOKEN_FILTERS_RAW_FIELDS=role,product,department
   ```

4. Добавить token-поле в Milvus schema:

   ```json
   {
     "name": "department_tokens",
     "dtype": "ARRAY",
     "element_type": "VARCHAR",
     "max_capacity": 32,
     "max_length": 255,
     "nullable": true
   }
   ```

5. Добавить token-поле в OpenSearch mapping:

   ```json
   "department_tokens": {
     "type": "keyword"
   }
   ```

6. Добавить поле в API DTO, если клиент должен передавать его как фильтр.
7. Прокинуть поле через service pack в Redis.
8. Убедиться, что `HybridSearchOrchestrator` читает его из pack через `settings.token_filters.raw_fields`.
9. Пересоздать OpenSearch index и Milvus collection, чтобы новые token-поля появились в индексах.
10. Добавить unit-тесты:
    - токенизация raw-поля;
    - runtime-нормализация;
    - OpenSearch filter clauses;
    - Milvus filter expression;
    - ingestion enrichment;
    - queue pack propagation.

## 20. Важные инварианты

1. Raw-поля сохраняются как есть.
2. Token-поля используются для фильтрации.
3. Ingestion split-ит raw-строку по `TOKEN_FILTERS_RAW_SEPARATOR`.
4. Runtime API-фильтры не split-ятся по `TOKEN_FILTERS_RAW_SEPARATOR`.
5. Внутри одного поля действует `OR`.
6. Между разными полями действует `AND`.
7. OpenSearch должен хранить token-поля как массивы keyword-значений.
8. Milvus должен хранить token-поля как `ARRAY[VARCHAR]`.
9. Presearch не фильтруется по token-фильтрам.
10. Cache key обязан учитывать token-фильтры.
11. После изменения схем Milvus/OpenSearch требуется rebuild/reindex.

## 21. Тестовое покрытие

Для реализации добавлены и/или обновлены тесты по следующим направлениям:

```text
tests/unit/test_token_filters.py
```

Проверяет:

- нормализацию raw-строк;
- удаление дублей;
- запрет частичного совпадения;
- построение token field names;
- runtime-нормализацию API-фильтров;
- построение OpenSearch clauses;
- построение Milvus expressions;
- детерминированность cache key.

```text
tests/unit/test_prepare_dataframe_tokens.py
```

Проверяет, что `prepare_dataframe` добавляет `role_tokens` и `product_tokens`.

```text
tests/unit/test_open_search_serialization.py
```

Проверяет, что OpenSearch adapter сохраняет `list[str]` как массив строк для keyword-полей.

```text
tests/unit/test_milvus_array_support.py
```

Проверяет поддержку `ARRAY[VARCHAR]` в Milvus-схеме и coercion массива.

```text
tests/unit/test_hybrid_search_service_filters.py
```

Проверяет, что `role` и `product` попадают в Redis pack задачи поиска.

```text
tests/unit/test_orchestrator_filter_queries.py
```

Проверяет, что OpenSearch lexical branch получает term-фильтры, а presearch не получает token-фильтры.

```text
tests/unit/test_orchestrator_presearch_pipeline.py
```

Проверяет:

- cache key зависит от фильтров;
- Milvus dense search получает `filter_expr`;
- presearch-результат инжектится в итоговую выдачу даже при наличии фильтров.

## 22. Известные follow-up замечания

### 22.1. Исправить устаревшее ожидание в одном тесте Milvus filter expression

В актуальной реализации `build_milvus_token_filter_expr` формирует выражение через `ARRAY_CONTAINS(...)`.

Один из тестов может ожидать старый синтаксис вида:

```text
array_contains_any(role_tokens, ["врач"])
```

Ожидание нужно привести к актуальному builder-у.

Рекомендуемый подход: в тесте не хардкодить строку вручную, а строить expected через `build_milvus_token_filter_expr(...)`.

### 22.2. Обновить устаревший docstring в `token_filters.py`

В комментариях к OpenSearch builder может оставаться упоминание, что фильтры применяются и в presearch. После изменения логики presearch это уже неверно.

Нужно оставить описание только про основной lexical-поиск, а для presearch отдельно указать, что он не фильтруется.

### 22.3. Опционально усилить `update_all`

`split_by_source` может вернуть `None` для одного из источников, если после фильтрации/дедупликации источник отсутствует. Сейчас стоит дополнительно защититься перед вызовом `prepare_dataframe`, чтобы updater не падал на пустом источнике.

### 22.4. Уточнить поведение prelaunch export

Если экспортируемый prelaunch-файл должен быть human-readable snapshot, текущая логика нормальна: raw-поля остаются, token-поля могут не попадать в экспортируемый порядок колонок.

Если экспорт должен быть полным техническим snapshot с token-полями, нужно добавить `role_tokens` и `product_tokens` в порядок экспортируемых колонок или отдельно документировать, что token-поля при повторной загрузке будут пересозданы из raw-полей.
