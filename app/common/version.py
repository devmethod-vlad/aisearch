from pathlib import Path

import toml


def get_app_info() -> dict:
    """Получение версии приложения"""
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parents[2]
    pyproject_path = project_root / "pyproject.toml"

    if not pyproject_path.exists():
        raise FileNotFoundError(f"Файл {pyproject_path} не найден.")

    with open(pyproject_path, encoding="utf-8") as f:
        info: dict = toml.load(f)

    return {
        "name": info["tool"]["poetry"]["name"],
        "version": info["tool"]["poetry"]["version"],
        "description": info["tool"]["poetry"]["description"],
    }
