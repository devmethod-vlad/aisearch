from pathlib import Path

import pandas as pd

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


def convert_parquet_to_xlsx(
    input_files: list[str],
    output_file: str | None = None,
    sheet_name: str = "Data",
    column_mapping: dict[str, str] | None = None,
    columns: list[str] | None = None,
    drop_duplicates: bool = True,
    duplicates_subset: list[str] | None = None,
) -> str:
    if not input_files:
        raise ValueError("Список входных файлов не может быть пустым")

    input_paths: list[Path] = []
    for input_file in input_files:
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Файл {input_file} не найден")
        if not input_path.suffix == ".parquet":
            raise ValueError(f"Файл должен иметь расширение .parquet: {input_file}")
        input_paths.append(input_path)

    if output_file is None:
        output_file = str(input_paths[0].with_suffix(".xlsx"))

    output_path = Path(output_file)

    dataframes: list[pd.DataFrame] = []

    for input_path in input_paths:
        df = pd.read_parquet(input_path)
        df = df.astype(str).replace("nan", "").replace("None", "").replace("<NA>", "")
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
                    error_msg += f"  Отсутствуют колонки: {sorted(diff_missing)}\n"
                if diff_extra:
                    error_msg += f"  Лишние колонки: {sorted(diff_extra)}"
                raise ValueError(error_msg)

    combined_df = (
        pd.concat(dataframes, ignore_index=True)
        if len(dataframes) > 1
        else dataframes[0]
    )

    if column_mapping:
        existing_columns = set(combined_df.columns)
        mapping_keys = set(column_mapping.keys())
        missing_columns = mapping_keys - existing_columns
        if missing_columns:
            raise ValueError(
                f"Колонки для переименования отсутствуют в данных: {sorted(missing_columns)}"
            )
        combined_df = combined_df.rename(columns=column_mapping)

    if columns:
        missing_columns = set(columns) - set(combined_df.columns)
        if missing_columns:
            raise ValueError(
                f"Указанные колонки отсутствуют в данных: {sorted(missing_columns)}"
            )
        combined_df = combined_df[columns]

    if drop_duplicates:
        if duplicates_subset:
            missing_subset = set(duplicates_subset) - set(combined_df.columns)
            if missing_subset:
                raise ValueError(
                    f"Колонки для проверки дубликатов отсутствуют: {sorted(missing_subset)}"
                )
            combined_df = combined_df.drop_duplicates(subset=duplicates_subset)
        else:
            combined_df = combined_df.drop_duplicates()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
        combined_df.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False,
        )

    if not output_path.exists():
        raise ValueError(f"Выходной файл не был создан: {output_file}")

    return str(output_path)


if __name__ == "__main__":

    convert_parquet_to_xlsx(
        input_files=["kb_default.parquet"],
        output_file="combined_kb.xlsx",
        sheet_name="База знаний",
        column_mapping={
            "source": COL_SOURCE,
            "ext_id": COL_ID,
            "page_id": COL_PAGE_ID,
            "actual": COL_ACTUAL,
            "second_line": COL_SECOND_LINE,
            "role": COL_ROLE,
            "product": COL_PRODUCT,
            "space": COL_SPACE,
            "component": COL_COMPONENT,
            "question_md": COL_Q_MD,
            "question": COL_Q,
            "analysis_md": COL_A_ERR_MD,
            "analysis": COL_A_ERR,
            "answer_md": COL_A_MD,
            "answer": COL_A,
            "for_user": COL_FOR_USER,
        },
        columns=[
            COL_SOURCE,
            COL_ID,
            COL_PAGE_ID,
            COL_ACTUAL,
            COL_SECOND_LINE,
            COL_ROLE,
            COL_PRODUCT,
            COL_SPACE,
            COL_COMPONENT,
            COL_Q_MD,
            COL_Q,
            COL_A_ERR_MD,
            COL_A_ERR,
            COL_A_MD,
            COL_A,
            COL_FOR_USER,
        ],
        drop_duplicates=True,
        duplicates_subset=[COL_ID, COL_PAGE_ID, COL_Q],
    )
