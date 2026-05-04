from app.infrastructure.adapters.open_search import OpenSearchAdapter


def test_coerce_for_mapping_keeps_string_arrays_for_keyword_fields() -> None:
    adapter = OpenSearchAdapter.__new__(OpenSearchAdapter)
    mapping = {
        "role_tokens": "keyword",
        "component_tokens": "keyword",
        "role": "keyword",
        "row_idx": "integer",
    }

    value = adapter._coerce_for_mapping("role_tokens", ["врач", "медсестра"], mapping)
    assert value == ["врач", "медсестра"]
    component_value = adapter._coerce_for_mapping(
        "component_tokens", ["назначения", "расписания"], mapping
    )
    assert component_value == ["назначения", "расписания"]

    assert adapter._coerce_for_mapping("role", "Врач", mapping) == "Врач"
    assert adapter._coerce_for_mapping("row_idx", "3", mapping) == 3
