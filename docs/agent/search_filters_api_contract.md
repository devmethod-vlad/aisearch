# Search filters API contract

## 1. Цель изменения API
Перевести внешний контракт поиска с плоских полей фильтров на единый вложенный объект `filters`, чтобы явно разделить `array_filters` и `exact_filters`.

## 2. Старый формат
```json
{
  "query": "как оформить назначение",
  "top_k": 5,
  "role": ["Врач"],
  "actual": "Да"
}
```

## 3. Новый формат
```json
{
  "query": "как оформить назначение",
  "top_k": 5,
  "filters": {
    "array_filters": {
      "role": ["Врач", "Медсестра"],
      "component": ["ЕМИАС"]
    },
    "exact_filters": {
      "actual": "Да",
      "second_line": "Да"
    }
  }
}
```

`filters`, `filters.array_filters` и `filters.exact_filters` — необязательные объекты.

## 4. Семантика array_filters
- Поля: `role`, `product`, `component`.
- Значения: массивы строк.
- Логика: OR внутри поля, AND между полями.
- Внешнее имя: `array_filters`.
- Внутренний runtime-термин: `token_filters` (после нормализации).

## 5. Семантика exact_filters
- Поля: `source`, `actual`, `second_line`.
- Значения: одиночные строки.
- Логика: AND между полями.
- Без split, только нормализация строки.

## 6. Router mapping
Router извлекает `body.filters.*` и передаёт в сервис:
- `array_filters = filters.array_filters.model_dump(exclude_none=True)`
- `exact_filters = filters.exact_filters.model_dump(exclude_none=True)`

Если блоков нет — передаются пустые словари.

## 7. Redis pack
Pack сохраняется так:
```json
{
  "type": "search",
  "query": "...",
  "top_k": 5,
  "array_filters": {},
  "exact_filters": {}
}
```

## 8. Orchestrator
- Читает `array_filters` из `pack["array_filters"]`.
- Читает `exact_filters` из `pack["exact_filters"]`.
- Нормализует и применяет оба типа фильтров в OpenSearch и Milvus через AND.
- Cache key учитывает оба типа фильтров: `tokens:<...>;exact:<...>`.

## 9. Совместимость и миграция фронтенда
Плоские поля `role/product/component/source/actual/second_line` больше не являются основным API-контрактом и должны быть заменены на nested `filters`.

## 10. Presearch
Presearch остаётся без фильтрации и не использует ни `array_filters`, ни `exact_filters`.


## 11. Runtime-параметры
Дополнительно контракт поддерживает `search_use_cache`, `show_intermediate_results` и `presearch.field`.
- `search_use_cache=false` отключает только cache read, но не cache write.
- `show_intermediate_results=true` возвращает `dense/lex/ce` и принудительно пропускает cache read.
- `presearch` включается только при наличии объекта с непустым `field`.
