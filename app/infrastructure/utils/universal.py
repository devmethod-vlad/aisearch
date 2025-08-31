import re
import typing as tp
from io import BytesIO
from urllib.parse import urlparse

import numpy as np
import pandas as pd


def normalize_string(el: tp.Any) -> str:
    """Нормализация строки"""
    if not isinstance(el, str):
        if np.isnan(el) or el is None:
            return ""
        return normalize_string(str(el))
    el = re.sub(r"\s+", " ", el).strip()
    return el


def is_absolute_url(url: str) -> bool:
    """Проверка строки на полный URL"""
    parsed_url = urlparse(url)
    return bool(parsed_url.scheme) and bool(parsed_url.netloc)


def cast_to_int(value: tp.Any) -> int:
    """Приведение элемента к целому числу"""
    if (
        isinstance(value, str)
        and value.isdigit()
        or isinstance(value, (int, float))
        and not np.isnan(value)
    ):
        return int(value)
    else:
        return 0


def generate_row_order_mapping(original: str, target: str, separator: str = ";") -> list[int]:
    """Генерация списка номеров позиций для преобразования порядка подстрок"""
    original_subrows = original.split(separator)
    target_subrows = target.split(separator)

    row_positions = {row: i for i, row in enumerate(original_subrows)}

    result = []
    for row in target_subrows:
        if row in row_positions:
            result.append(row_positions[row])
        else:
            raise ValueError(f"Строка '{row}' не найдена в оригинале")

    return result


def convert_dataframe_to_xlsx_file(df: pd.DataFrame) -> BytesIO:
    """Перевод объекта pandas DataFrame в BytesIO xlsx файл"""
    excel_file = BytesIO()

    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)

    excel_file.seek(0)
    return excel_file


def convert_dataframe_to_list_of_tuples(
    df: pd.DataFrame, columns: list[str] | None = None
) -> list[tuple]:
    """Перевод объекта pandas DataFrame в список кортежей"""
    return [
        tuple(d.values()) for d in df[columns if columns else df.columns].to_dict(orient="records")
    ]
