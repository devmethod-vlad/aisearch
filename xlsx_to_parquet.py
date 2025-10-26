from pathlib import Path
import pandas as pd
import typing as tp

COL_SOURCE = "Источник"
COL_ID = "ID"
COL_PAGE_ID = "ID страницы"
COL_ACTUAL = "Актуально"
COL_SECOND_LINE = "2 линия"
COL_ROLE = "Роль"
COL_PRODUCT = "Продукт"
COL_SPACE = "Пространство"
COL_COMPONENT = "Компонент"
COL_Q_MD = "Вопрос (markdown)"
COL_Q = "Вопрос (clean)"
COL_A_ERR_MD = "Анализ ошибки (markdown)"
COL_A_ERR = "Анализ ошибки (clean)"
COL_A_MD = "Ответ (markdown)"
COL_A = "Ответ (clean)"
COL_FOR_USER = "Для пользователя"


def convert_xlsx_to_parquet(
    input_files: list[str],
    output_file: str = None,
    sheet_name: str = None,
    row_idx_field: bool = True,
    column_mapping: tp.Optional[dict[str, str]] = None,
    columns: tp.Optional[list[str]] = None,
) -> str:
    """Конвертация нескольких XLSX файлов в один Parquet формат"""

    if not input_files:
        raise ValueError("Список входных файлов не может быть пустым")

    input_paths = []
    for input_file in input_files:
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Файл {input_file} не найден")
        input_paths.append(input_path)

    if output_file is None:
        output_file = input_paths[0].with_suffix(".parquet")

    output_path = Path(output_file)

    dataframes: list[pd.DataFrame] = []

    for input_file in input_files:
        df: pd.DataFrame = (
            (
                pd.read_excel(input_file, sheet_name=sheet_name)
                if sheet_name
                else pd.read_excel(input_file)
            )
            .fillna("")
            .astype(str)
        )

        if column_mapping:
            df = df.rename(columns=column_mapping)

        dataframes.append(df)

    if dataframes:
        first_columns = set(dataframes[0].columns)
        for i, df in enumerate(dataframes[1:], 1):
            current_columns = set(df.columns)
            if first_columns != current_columns:
                diff_missing = first_columns - current_columns
                diff_extra = current_columns - first_columns
                error_msg = f"Файл {input_files[i]} имеет отличающиеся колонки:\n"
                if diff_missing:
                    error_msg += f"  Отсутствуют колонки: {diff_missing}\n"
                if diff_extra:
                    error_msg += f"  Лишние колонки: {diff_extra}"
                raise ValueError(error_msg)

    if len(dataframes) > 1:
        combined_df = pd.concat(dataframes, ignore_index=True)
    else:
        combined_df = dataframes[0]

    if columns:
        missing_columns = set(columns) - set(combined_df.columns)
        if missing_columns:
            raise ValueError(
                f"Указанные колонки отсутствуют в данных: {missing_columns}"
            )
        combined_df = combined_df[columns]

    if row_idx_field:
        combined_df.insert(0, "row_idx", combined_df.index)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(output_file, index=False, engine="pyarrow")
    combined_df.to_excel("result.xlsx", index=False)
    return str(output_file)


if __name__ == "__main__":
    convert_xlsx_to_parquet(
        ["kb_default.xlsx", "vio_export.xlsx"],
        "kb_default.parquet",
        row_idx_field=True,
        column_mapping={
            COL_SOURCE: "source",
            COL_ID: "ext_id",
            COL_PAGE_ID: "page_id",
            COL_ACTUAL: "actual",
            COL_SECOND_LINE: "second_line",
            COL_ROLE: "role",
            COL_PRODUCT: "product",
            COL_SPACE: "space",
            COL_COMPONENT: "component",
            COL_Q_MD: "question_md",
            COL_Q: "question",
            COL_A_ERR_MD: "analysis_md",
            COL_A_ERR: "analysis",
            COL_A_MD: "answer_md",
            COL_A: "answer",
            COL_FOR_USER: "for_user",
        },
        columns=[
            "source",
            "ext_id",
            "page_id",
            "actual",
            "second_line",
            "role",
            "product",
            "space",
            "component",
            "question_md",
            "question",
            "analysis_md",
            "analysis",
            "answer_md",
            "answer",
            "for_user",
        ],
    )
