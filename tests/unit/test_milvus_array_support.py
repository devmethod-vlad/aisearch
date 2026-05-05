import json
from pathlib import Path

from pymilvus import DataType, FieldSchema

from app.infrastructure.storages.milvus import MilvusDatabase
from app.infrastructure.utils.milvus import load_schema_and_indexes_from_json


def test_load_schema_and_indexes_supports_array_varchar(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "fields": [
                    {
                        "name": "pk",
                        "dtype": "INT64",
                        "is_primary": True,
                        "auto_id": True,
                    },
                    {
                        "name": "role_tokens",
                        "dtype": "ARRAY",
                        "element_type": "VARCHAR",
                        "max_capacity": 32,
                        "max_length": 255,
                    },
                    {
                        "name": "component_tokens",
                        "dtype": "ARRAY",
                        "element_type": "VARCHAR",
                        "max_capacity": 32,
                        "max_length": 255,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    fields, _, _ = load_schema_and_indexes_from_json(str(schema_path))
    role_field = next(field for field in fields if field.name == "role_tokens")
    component_field = next(
        field for field in fields if field.name == "component_tokens"
    )
    assert role_field.dtype == DataType.ARRAY
    assert role_field.element_type == DataType.VARCHAR
    assert component_field.dtype == DataType.ARRAY
    assert component_field.element_type == DataType.VARCHAR


def test_milvus_coerce_array_value() -> None:
    db = MilvusDatabase.__new__(MilvusDatabase)
    field = FieldSchema(
        name="role_tokens",
        dtype=DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=32,
        max_length=255,
    )

    coerced, drop = db._coerce_field_value(
        {"role_tokens": field}, "role_tokens", ["врач", "медсестра"]
    )

    assert drop is False
    assert coerced == ["врач", "медсестра"]
