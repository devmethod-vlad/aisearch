from app.settings.config import TokenFiltersSettings


def test_token_filter_settings_parse_fields_from_env_string() -> None:
    settings = TokenFiltersSettings(
        env_separator=",",
        raw_fields="role, product , department",
        token_suffix="_tokens",
        raw_separator=";",
    )

    assert settings.raw_fields == ("role", "product", "department")
    assert settings.token_suffix == "_tokens"
    assert settings.raw_separator == ";"
