import typing as tp
from dataclasses import dataclass

"""Утилиты token-фильтрации для ingest и runtime-поиска.

Модуль объединяет два связанных этапа:
- ingestion: из raw-метаданных строятся дополнительные `*_tokens` поля;
- search runtime: входные API-фильтры нормализуются в те же токены, после чего
  из них собираются backend-специфичные условия для OpenSearch и Milvus.

Эти функции используются в `prepare_dataframe` (pre_launch/updater) и в
`HybridSearchOrchestrator` при формировании фильтров и cache key.
"""


@dataclass(frozen=True)
class MultiValueTokenConfig:
    """Конфиг token-фильтрации мультизначных raw-полей.

    Используется в двух контурах:
    1) ingestion (`prepare_dataframe`) — чтобы единообразно строить token-поля;
    2) runtime (`HybridSearchOrchestrator`) — чтобы одинаково трактовать
       входные фильтры API и генерировать backend-фильтры.
    """

    raw_fields: tuple[str, ...]
    token_suffix: str
    raw_separator: str

    def token_field(self, raw_field: str) -> str:
        """Возвращает имя token-поля для исходного raw-поля.

        Контракт: имя формируется детерминированно и без доступа к данным
        записи; это важно, чтобы ingest и search использовали одни и те же
        имена полей (`role` -> `role_tokens`, и т.д.).
        """
        return f"{raw_field}{self.token_suffix}"

    @property
    def token_fields(self) -> tuple[str, ...]:
        """Список всех token-полей, соответствующих `raw_fields`."""
        return tuple(self.token_field(field) for field in self.raw_fields)


@dataclass(frozen=True)
class NormalizedTokenFilters:
    """Нормализованные фильтры в терминах token-полей.

    Формат `by_token_field`: `{token_field: (token1, token2, ...)}`.
    Внутри одного поля набор значений трактуется как OR, а разные поля — как
    AND. Это представление затем конвертируется в OpenSearch/Milvus-запросы.
    """

    by_token_field: dict[str, tuple[str, ...]]

    def is_empty(self) -> bool:
        """Проверяет, есть ли реально применимые token-фильтры."""
        return not self.by_token_field

    def cache_key_part(self) -> str:
        """Строит детерминированный фрагмент cache key из фильтров.

        Используется в `HybridSearchOrchestrator.documents_search`, чтобы
        логически одинаковые фильтры всегда давали одинаковый ключ кеша
        независимо от порядка полей во входном словаре.
        """
        if not self.by_token_field:
            return "no_filters"

        parts: list[str] = []
        # Поля сортируем, чтобы ключ не зависел от порядка в dict.
        for field in sorted(self.by_token_field):
            tokens = ",".join(self.by_token_field[field])
            parts.append(f"{field}={tokens}")
        return "|".join(parts)


def normalize_token(value: tp.Any) -> str:
    """Нормализует единичный токен (`str` -> trim -> casefold).

    Это базовая примитивная нормализация, на которой держатся и построение
    token-полей при индексации, и нормализация API-фильтров при поиске.
    """
    return str(value).strip().casefold()


def tokenize_record_raw_value(
    raw_value: tp.Any,
    *,
    separator: str,
) -> list[str]:
    """Токенизирует raw metadata поле записи через raw-separator.

    Используется на этапе enrichment в ingestion-пайплайне (pre_launch/updater).
    Алгоритм: split -> trim/casefold -> удаление пустых -> dedup с сохранением
    порядка.

    Важно: это логика именно для raw metadata поля (обычно строка с
    разделителем `;`). Для API-фильтров применяется отдельная функция, где
    каждый элемент списка уже считается отдельным значением.
    """
    if raw_value is None:
        return []

    raw_text = str(raw_value)
    if raw_text == "":
        return []

    unique: dict[str, None] = {}
    for chunk in raw_text.split(separator):
        token = normalize_token(chunk)
        if token:
            unique.setdefault(token, None)

    return list(unique.keys())


def normalize_request_filter_values(
    raw_values: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Нормализует значения одного входного фильтра API.

    Важно: каждый элемент списка рассматривается как отдельное выбранное
    значение и НЕ split-ится повторно по `raw_separator`. API уже передаёт
    список выбранных значений, и дополнительный split здесь ломал бы семантику.
    """
    if not raw_values:
        return ()

    unique: dict[str, None] = {}
    for value in raw_values:
        token = normalize_token(value)
        if token:
            unique.setdefault(token, None)

    return tuple(unique.keys())


def build_token_fields_for_record(
    record: dict[str, tp.Any],
    *,
    config: MultiValueTokenConfig,
) -> dict[str, list[str]]:
    """Строит token-поля для одной записи на основе raw-полей из конфига.

    Применяется внутри `prepare_dataframe` при подготовке метаданных к
    индексации в OpenSearch/Milvus. Возвращает только token-поля, без изменения
    исходного словаря записи.
    """
    return {
        config.token_field(raw_field): tokenize_record_raw_value(
            record.get(raw_field), separator=config.raw_separator
        )
        for raw_field in config.raw_fields
    }


def enrich_records_with_token_fields(
    records: list[dict[str, tp.Any]],
    *,
    config: MultiValueTokenConfig,
) -> list[dict[str, tp.Any]]:
    """Возвращает новый список records с добавленными token-полями.

    Функция централизует enrichment для двух ingestion-путей:
    - `pre_launch.prepare_and_load_data`;
    - `UpdaterService.update_vio_base` / `update_kb_base`.
    Это гарантирует одинаковую токенизацию метаданных при полной и
    инкрементальной загрузке.
    """
    enriched: list[dict[str, tp.Any]] = []
    for row in records:
        item = dict(row)
        item.update(build_token_fields_for_record(item, config=config))
        enriched.append(item)
    return enriched


def normalize_request_token_filters(
    raw_filters: dict[str, list[str] | None],
    *,
    config: MultiValueTokenConfig,
) -> NormalizedTokenFilters:
    """Нормализует входные фильтры API в `token_field -> tuple[token]`.

    Используется в `HybridSearchOrchestrator.documents_search` до построения
    фильтров для OpenSearch/Milvus и перед генерацией cache key.
    """
    by_field: dict[str, tuple[str, ...]] = {}
    for raw_field in config.raw_fields:
        tokens = normalize_request_filter_values(raw_filters.get(raw_field))
        if tokens:
            # Группируем значения по token-полю: внутри поля будет OR.
            by_field[config.token_field(raw_field)] = tuple(tokens)
    return NormalizedTokenFilters(by_token_field=by_field)


def build_opensearch_token_filter_clauses(
    filters: NormalizedTokenFilters,
) -> list[dict[str, tp.Any]]:
    """Строит `bool.filter` clauses для OpenSearch.

    Семантика:
    - OR внутри одного token-поля (`should + minimum_should_match=1`);
    - AND между разными token-полями (каждая группа — отдельный element в
      `bool.filter`).

    Используется в `_os_candidates` для основного lexical-поиска.
    `_presearch_exact_match` намеренно не использует эти clauses: presearch
    остаётся unfiltered exact-match этапом по `presearch_field`, а
    token-фильтры применяются только в основном dense/lex pipeline.
    При этом cache key всё равно учитывает token-фильтры, чтобы не
    смешивать результаты запросов с разными фильтрами.
    """
    clauses: list[dict[str, tp.Any]] = []
    for token_field in sorted(filters.by_token_field):
        # Внутри поля разрешаем совпадение по любому из выбранных токенов.
        should = [{"term": {token_field: token}} for token in filters.by_token_field[token_field]]
        clauses.append({"bool": {"should": should, "minimum_should_match": 1}})
    return clauses


def _escape_milvus_string(value: str) -> str:
    """Экранирует строку для безопасной подстановки в Milvus expression."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def build_milvus_token_filter_expr(filters: NormalizedTokenFilters) -> str | None:
    """Строит Milvus expression с той же логикой OR/AND, что и в OpenSearch.

    Результат используется в `HybridSearchOrchestrator.documents_search` как
    `filter_expr` при `vector_db.search(...)`. При отсутствии фильтров
    возвращается `None`, чтобы не добавлять лишних ограничений.
    """
    groups: list[str] = []
    for token_field in sorted(filters.by_token_field):
        # В Milvus массив токенов проверяется через ARRAY_CONTAINS.
        terms = [
            f'ARRAY_CONTAINS({token_field}, "{_escape_milvus_string(token)}")'
            for token in filters.by_token_field[token_field]
        ]
        if len(terms) == 1:
            groups.append(terms[0])
        elif terms:
            # OR внутри поля.
            groups.append(f"({' OR '.join(terms)})")
    if not groups:
        return None
    # AND между полями.
    return " AND ".join(groups)
