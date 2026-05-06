import typing as tp
from dataclasses import dataclass


@dataclass(frozen=True)
class ExactFilterConfig:
    """Конфиг exact-фильтрации single-value metadata полей."""

    raw_fields: tuple[str, ...]
    field_suffix: str

    def filter_field(self, raw_field: str) -> str:
        """Возвращает служебное поле фильтрации для исходного поля."""
        return f"{raw_field}{self.field_suffix}"

    @property
    def filter_fields(self) -> tuple[str, ...]:
        """Возвращает список всех служебных полей exact-фильтрации."""
        return tuple(self.filter_field(field) for field in self.raw_fields)


@dataclass(frozen=True)
class NormalizedExactFilters:
    """Нормализованные exact-фильтры в виде filter_field -> value."""

    by_filter_field: dict[str, str]

    def is_empty(self) -> bool:
        """Проверяет, есть ли применимые exact-фильтры."""
        return not self.by_filter_field

    def cache_key_part(self) -> str:
        """Строит детерминированный фрагмент cache key для exact-фильтров."""
        if not self.by_filter_field:
            return "no_exact_filters"
        return "|".join(
            f"{field}={self.by_filter_field[field]}"
            for field in sorted(self.by_filter_field)
        )


def normalize_exact_value(value: tp.Any) -> str:
    """Нормализует значение exact-фильтра (trim + casefold без split)."""
    return str(value).strip().casefold()


def build_exact_fields_for_record(
    record: dict[str, tp.Any],
    *,
    config: ExactFilterConfig,
) -> dict[str, str]:
    """Строит служебные *_filter поля для одной записи."""
    fields: dict[str, str] = {}
    for raw_field in config.raw_fields:
        fields[config.filter_field(raw_field)] = normalize_exact_value(record.get(raw_field, ""))
    return fields


def enrich_records_with_exact_filter_fields(
    records: list[dict[str, tp.Any]],
    *,
    config: ExactFilterConfig,
) -> list[dict[str, tp.Any]]:
    """Возвращает новый список records с добавленными *_filter полями."""
    enriched: list[dict[str, tp.Any]] = []
    for row in records:
        item = dict(row)
        item.update(build_exact_fields_for_record(item, config=config))
        enriched.append(item)
    return enriched


def normalize_request_exact_filters(
    raw_filters: dict[str, tp.Any],
    *,
    config: ExactFilterConfig,
) -> NormalizedExactFilters:
    """Нормализует raw exact-фильтры запроса в filter_field -> value."""
    by_field: dict[str, str] = {}
    for raw_field in config.raw_fields:
        raw_value = raw_filters.get(raw_field)
        if raw_value is None:
            continue
        normalized = normalize_exact_value(raw_value)
        if normalized:
            by_field[config.filter_field(raw_field)] = normalized
    return NormalizedExactFilters(by_filter_field=by_field)


def build_opensearch_exact_filter_clauses(
    filters: NormalizedExactFilters,
) -> list[dict[str, tp.Any]]:
    """Строит term-clause exact-фильтрации для OpenSearch bool.filter."""
    return [
        {"term": {field: filters.by_filter_field[field]}}
        for field in sorted(filters.by_filter_field)
    ]


def _escape_milvus_string(value: str) -> str:
    """Экранирует спецсимволы для Milvus expression-строк."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def build_milvus_exact_filter_expr(filters: NormalizedExactFilters) -> str | None:
    """Строит exact выражение Milvus по `==` и AND между полями."""
    parts = [
        f'{field} == "{_escape_milvus_string(filters.by_filter_field[field])}"'
        for field in sorted(filters.by_filter_field)
    ]
    return " AND ".join(parts) if parts else None


def combine_milvus_filter_exprs(*exprs: str | None) -> str | None:
    """Объединяет непустые Milvus expression через AND."""
    valid = [expr for expr in exprs if expr and expr.strip()]
    return " AND ".join(valid) if valid else None
