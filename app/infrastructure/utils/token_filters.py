import typing as tp
from dataclasses import dataclass


@dataclass(frozen=True)
class MultiValueTokenConfig:
    """Единый конфиг для token-полей мультизначных метаданных."""

    raw_fields: tuple[str, ...] = ("role", "product")
    token_suffix: str = "_tokens"
    raw_separator: str = ";"

    def token_field(self, raw_field: str) -> str:
        return f"{raw_field}{self.token_suffix}"

    @property
    def token_fields(self) -> tuple[str, ...]:
        return tuple(self.token_field(field) for field in self.raw_fields)


TOKEN_FILTER_CONFIG = MultiValueTokenConfig()


@dataclass(frozen=True)
class NormalizedTokenFilters:
    """Нормализованные фильтры в терминах token-полей."""

    by_token_field: dict[str, tuple[str, ...]]

    def is_empty(self) -> bool:
        return not self.by_token_field

    def cache_key_part(self) -> str:
        if not self.by_token_field:
            return "no_filters"

        parts: list[str] = []
        for field in sorted(self.by_token_field):
            tokens = ",".join(self.by_token_field[field])
            parts.append(f"{field}={tokens}")
        return "|".join(parts)


def normalize_token(value: tp.Any) -> str:
    """Нормализует единичный токен: trim + casefold."""
    return str(value).strip().casefold()


def tokenize_multi_value(
    raw_value: tp.Any,
    *,
    separator: str | None = None,
) -> list[str]:
    """Разбивает сырое поле на нормализованные уникальные токены.

    Алгоритм: split -> trim/casefold -> удаление пустых -> dedup с сохранением порядка.
    """
    if raw_value is None:
        return []

    sep = separator or TOKEN_FILTER_CONFIG.raw_separator
    raw_text = str(raw_value)
    if raw_text == "":
        return []

    unique: dict[str, None] = {}
    for chunk in raw_text.split(sep):
        token = normalize_token(chunk)
        if token:
            unique.setdefault(token, None)

    return list(unique.keys())


def build_token_fields_for_record(
    record: dict[str, tp.Any],
    *,
    config: MultiValueTokenConfig = TOKEN_FILTER_CONFIG,
) -> dict[str, list[str]]:
    """Строит token-поля для одной записи на основе raw-полей из конфига."""
    return {
        config.token_field(raw_field): tokenize_multi_value(
            record.get(raw_field), separator=config.raw_separator
        )
        for raw_field in config.raw_fields
    }


def enrich_records_with_token_fields(
    records: list[dict[str, tp.Any]],
    *,
    config: MultiValueTokenConfig = TOKEN_FILTER_CONFIG,
) -> list[dict[str, tp.Any]]:
    """Возвращает новый список records с добавленными token-полями."""
    enriched: list[dict[str, tp.Any]] = []
    for row in records:
        item = dict(row)
        item.update(build_token_fields_for_record(item, config=config))
        enriched.append(item)
    return enriched


def normalize_request_token_filters(
    raw_filters: dict[str, tp.Any],
    *,
    config: MultiValueTokenConfig = TOKEN_FILTER_CONFIG,
) -> NormalizedTokenFilters:
    """Нормализует входные фильтры API в структуру token_field -> tuple[token]."""
    by_field: dict[str, tuple[str, ...]] = {}
    for raw_field in config.raw_fields:
        tokens = tokenize_multi_value(
            raw_filters.get(raw_field), separator=config.raw_separator
        )
        if tokens:
            by_field[config.token_field(raw_field)] = tuple(tokens)
    return NormalizedTokenFilters(by_token_field=by_field)


def build_opensearch_token_filter_clauses(
    filters: NormalizedTokenFilters,
) -> list[dict[str, tp.Any]]:
    """Строит bool.filter clauses для OpenSearch term-поиска по token-полям."""
    clauses: list[dict[str, tp.Any]] = []
    for token_field in sorted(filters.by_token_field):
        for token in filters.by_token_field[token_field]:
            clauses.append({"term": {token_field: token}})
    return clauses


def _escape_milvus_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def build_milvus_token_filter_expr(filters: NormalizedTokenFilters) -> str | None:
    """Строит Milvus filter expression через ARRAY_CONTAINS(...)."""
    terms: list[str] = []
    for token_field in sorted(filters.by_token_field):
        for token in filters.by_token_field[token_field]:
            terms.append(f'ARRAY_CONTAINS({token_field}, "{_escape_milvus_string(token)}")')
    if not terms:
        return None
    return " AND ".join(terms)
