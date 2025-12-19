import json
import typing as tp
from dataclasses import dataclass

from pymilvus import DataType, FieldSchema

DTYPE_MAP = {
    "BOOL": DataType.BOOL,
    "INT8": DataType.INT8,
    "INT16": DataType.INT16,
    "INT32": DataType.INT32,
    "INT64": DataType.INT64,
    "FLOAT": DataType.FLOAT,
    "DOUBLE": DataType.DOUBLE,
    "VARCHAR": DataType.VARCHAR,
    "FLOAT_VECTOR": DataType.FLOAT_VECTOR,
    "BINARY_VECTOR": DataType.BINARY_VECTOR,
}


@dataclass(frozen=True)
class IndexSpec:
    field_name: str
    index_type: str
    metric_type: str | None = None
    params: dict[str, tp.Any] | None = None


def load_schema_and_indexes_from_json(
    path: str,
) -> tuple[list[FieldSchema], list[IndexSpec], dict[str, dict[str, tp.Any]]]:
    """Возвращает (fields, indexes, search_params_by_field).
    - fields: список FieldSchema для CollectionSchema
    - indexes: список IndexSpec для create_index
    - search_params_by_field: мапа { field_name -> dict с поисковыми параметрами }
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if "fields" not in data or not isinstance(data["fields"], list):
        raise ValueError(f"Некорректный формат JSON схемы (нет fields): {path}")

    fields: list[FieldSchema] = []
    for raw_field in data["fields"]:
        dtype_str = raw_field.get("dtype")
        if not dtype_str:
            raise ValueError(f"Поле без dtype: {raw_field}")
        dtype = DTYPE_MAP.get(dtype_str)
        if dtype is None:
            raise ValueError(
                f"Неподдерживаемый dtype '{dtype_str}' в поле: {raw_field}"
            )

        kwargs: dict[str, tp.Any] = {
            "name": raw_field["name"],
            "dtype": dtype,
            "is_primary": raw_field.get("is_primary", False),
            "auto_id": raw_field.get("auto_id", False),
            "description": raw_field.get("description", ""),
            "nullable": raw_field.get("nullable", False),
        }
        if dtype == DataType.VARCHAR:
            kwargs["max_length"] = raw_field.get("max_length", 255)
        elif dtype in (DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR):
            kwargs["dim"] = raw_field.get("dim")
        fields.append(FieldSchema(**kwargs))

    # Индексы (опционально)
    index_specs: list[IndexSpec] = []
    for raw_idx in data.get("indexes", []) or []:
        field_name = raw_idx.get("field_name")
        index_type = raw_idx.get("index_type")
        metric_type = raw_idx.get("metric_type")
        params = raw_idx.get("params") or None
        if not field_name or not index_type:
            raise ValueError(
                f"Некорректный индекс (нужны field_name и index_type): {raw_idx}"
            )
        index_specs.append(
            IndexSpec(
                field_name=field_name,
                index_type=index_type,
                metric_type=metric_type,
                params=params,
            )
        )

    # Поисковые параметры по полям (опционально)
    search_params_by_field: dict[str, dict[str, tp.Any]] = (
        data.get("search_params") or {}
    )

    return fields, index_specs, search_params_by_field
