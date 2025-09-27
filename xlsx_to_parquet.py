from pathlib import Path

import pandas as pd

COL_SOURCE = "Источник"
COL_ID = "ID"
COL_PAGE_ID = "ID страницы"
COL_ROLE = "Роль"
COL_COMPONENT = "Компонент"
COL_Q = "Вопрос (clean)"
COL_A_ERR = "Анализ ошибки (clean)"
COL_A = "Ответ (clean)"


def convert_xlsx_to_parquet(
    input_file: str,
    output_file: str = None,
    sheet_name: str = None,
    row_idx_field: bool = True,
    column_mapping: dict[str, str] | None = None,
    columns: list[str] | None = None,
) -> str:
    """Конвертация XLSX файла в Parquet формат"""
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Файл {input_file} не найден")

    if output_file is None:
        output_file = input_path.with_suffix(".parquet")  # type: ignore

    output_path = Path(output_file)

    df = (
        pd.read_excel(input_file, sheet_name=sheet_name)
        if sheet_name
        else pd.read_excel(input_file)
    )

    if column_mapping:
        df = df.rename(columns=column_mapping)

    if columns:
        df = df[columns]

    if row_idx_field:
        df.insert(0, "row_idx", df.index)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False, engine="pyarrow")

    return str(output_file)


if __name__ == "__main__":
    convert_xlsx_to_parquet(
        "kb_default.xlsx",
        "kb_default.parquet",
        row_idx_field=True,
        column_mapping={
            COL_SOURCE: "source",
            COL_ID: "ext_id",
            COL_PAGE_ID: "page_id",
            COL_ROLE: "role",
            COL_COMPONENT: "component",
            COL_Q: "question",
            COL_A_ERR: "analysis",
            COL_A: "answer",
        },
        columns=[
            "source",
            "ext_id",
            "page_id",
            "role",
            "component",
            "question",
            "analysis",
            "answer",
        ],
    )
