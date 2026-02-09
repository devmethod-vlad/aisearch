import json
from pathlib import Path

import pandas as pd


def load_field_mapping(config_path: str | Path) -> dict[str, str]:
    """Загружает маппинг полей из JSON файла"""
    config_path = Path(config_path)

    try:
        with open(config_path, encoding="utf-8") as f:
            field_mapping = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Ошибка чтения JSON конфига: {e}")

    if not isinstance(field_mapping, dict):
        raise ValueError(
            f"Некорректный формат конфига: ожидается dict, получен {type(field_mapping)}"
        )

    return field_mapping


def reorder_columns_by_mapping(
    df: pd.DataFrame, mapping_config_path: str | Path, id_column: str
) -> pd.DataFrame:
    """Упорядочивает колонки DataFrame согласно порядку в field_mapping.json."""
    field_mapping = load_field_mapping(mapping_config_path)

    target_columns = list(field_mapping.values())
    existing_columns = [col for col in target_columns if col in df.columns]

    if "row_idx" in df.columns and "row_idx" not in existing_columns:
        existing_columns.append("row_idx")

    required_columns = ["source", id_column, "question", "answer", "modified_at"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing_columns}")

    return df[existing_columns]


def rename_dataframe(df: pd.DataFrame, config_path: str | Path) -> pd.DataFrame:
    """Переименовывает поля DataFrame по маппингу из конфига"""
    field_mapping = load_field_mapping(config_path)
    df = df.copy()

    available_mapping = {
        src: dst for src, dst in field_mapping.items() if src in df.columns
    }

    if available_mapping:
        df.rename(columns=available_mapping, inplace=True)

    return df


def validate_dataframe(
    df: pd.DataFrame, config_path: str | Path, id_column: str
) -> pd.DataFrame:
    """Валидирует DataFrame: проверяет, что все поля из маппинга присутствуют.
    Предполагается, что DataFrame уже переименован.
    """
    field_mapping = load_field_mapping(config_path)

    if df.empty:
        raise ValueError("DataFrame пуст")

    target_fields = list(field_mapping.values())
    missing_fields = [
        target_field for target_field in target_fields if target_field not in df.columns
    ]

    if missing_fields:
        raise ValueError(
            f"Отсутствуют поля в DataFrame: {missing_fields}. "
            f"Доступные поля: {list(df.columns)}"
        )

    if id_column not in df.columns:
        raise ValueError(f"Отсутствует поле уникального идентификатора: {id_column}")

    if df[id_column].isna().any():
        raise ValueError(f"Обнаружены записи без {id_column}")

    return df


def load_files_from_directory(
    files_dir: str | Path,
    file_extensions: tuple[str, ...] = (".parquet", ".xlsx", ".xls"),
) -> list[tuple[Path, pd.DataFrame]]:
    """Загружает все файлы из директории, возвращает список (путь, DataFrame)"""
    files_dir = Path(files_dir)

    if not files_dir.exists():
        raise ValueError(f"Директория не существует: {files_dir}")

    valid_files: list[Path] = []
    for ext in file_extensions:
        valid_files.extend(files_dir.glob(f"*{ext}"))

    if not valid_files:
        raise ValueError(
            f"Файлы с расширениями {file_extensions} не найдены в {files_dir}"
        )

    loaded_files = []
    for file_path in valid_files:
        if file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
        else:  # .xlsx или .xls
            df = pd.read_excel(file_path)

        loaded_files.append((file_path, df))

    return loaded_files


def combine_validated_sources(
    dataframes: list[pd.DataFrame], id_column: str
) -> pd.DataFrame:
    """Объединяет провалидированные источники.

    Процесс:
    1. Выстраивает колонки в алфавитном порядке (чтобы избежать ошибок при сложении)
    2. Складывает датафреймы (их может быть много)
    3. Удаляет дубликаты по id_column
    """
    if not dataframes:
        raise ValueError("Нет DataFrame для объединения")

    first_columns = set(dataframes[0].columns)
    for i, df in enumerate(dataframes[1:], 1):
        current_columns = set(df.columns)
        if first_columns != current_columns:
            missing_cols = first_columns - current_columns
            extra_cols = current_columns - first_columns
            raise ValueError(
                f"DataFrame {i} имеет отличную структуру от первого DataFrame. "
                f"Отсутствующие колонки: {missing_cols}. "
                f"Лишние колонки: {extra_cols}"
            )

    aligned_dfs: list[pd.DataFrame] = []
    for df in dataframes:
        sorted_columns = sorted(df.columns)
        aligned_df = df[sorted_columns].copy()
        aligned_dfs.append(aligned_df)

    combined_df = pd.concat(aligned_dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=[id_column], keep="last")

    return combined_df


def prepare_dataframe(
    df: pd.DataFrame, id_column: str
) -> tuple[list[str], list[dict], pd.DataFrame]:
    """Унифицированная очистка/фильтрация данных для pre_launch и updater."""
    df = df.copy()

    if "page_id" in df.columns:
        df["page_id"] = (
            pd.to_numeric(df["page_id"], errors="coerce")
            .astype("Int64")
            .astype("string")
            .fillna("")
        )

    df = df.astype("string").fillna("")

    df[id_column] = df[id_column].astype(str).str.strip()
    df["source"] = df["source"].astype(str).str.strip()

    # базовая фильтрация
    base_mask = df["question"].notna() & (df["question"].astype(str).str.len() > 0)
    df = df[base_mask]

    # source
    source_lower = df["source"].astype(str).str.lower().str.casefold().str.strip()

    vio_mask = source_lower == "вио"
    tp_mask = source_lower == "тп"

    vio_df = df[vio_mask].copy()
    tp_df = df[tp_mask].copy()

    # фильтры ВиО
    if not vio_df.empty:
        vio_df = vio_df[~vio_df["actual"].astype(str).str.lower().str.contains("нет")]
        space_norm = vio_df["space"].astype("string").str.strip().str.casefold()
        mask = space_norm.notna() & space_norm.ne("") & space_norm.ne("не распределено")
        vio_df = vio_df[mask]

    # фильтры ТП
    if not tp_df.empty:
        tp_df = tp_df[~tp_df["actual"].astype(str).str.lower().str.contains("нет")]

    df_final = pd.concat([vio_df, tp_df], ignore_index=True)
    df_final["row_idx"] = range(len(df_final))

    documents = df_final["question"].astype(str).tolist()
    metadata = df_final.to_dict(orient="records")

    return documents, metadata, df_final


def dedup_by_question_any(
    df: pd.DataFrame,
    question_col: str = "question",
    source_col: str = "source",
) -> pd.DataFrame:
    """Удаляет дубликаты знаний по вопросу с учетом приоритета источников:
    1. Если есть записи из 'ТП' → оставляем ВСЕ записи 'ТП'
    2. Если нет записей из 'ТП' → оставляем ПЕРВУЮ запись из 'ВиО' (удаляем дубликаты 'ВиО')
    3. Если есть и 'ТП', и 'ВиО' → оставляем только записи 'ТП'
    """
    df = df.copy()

    # Нормализуем вопрос
    q_norm = df[question_col].astype("string").str.strip().str.casefold()

    # Обработка NaN
    na = q_norm.isna()
    if na.any():
        q_norm = q_norm.where(~na, "__NA__" + df.index[na].astype(str))

    # Нормализуем источник
    source_norm = df[source_col].astype("string").str.strip().str.casefold()

    # Добавляем нормализованные столбцы
    df = df.assign(_q_norm=q_norm, _source_norm=source_norm)

    # Группируем по вопросу
    result_rows = []

    for _, group in df.groupby("_q_norm"):
        # Определяем маски для каждой группы отдельно
        tp_mask = group["_source_norm"] == "тп"
        vio_mask = group["_source_norm"] == "вио"

        # Разделяем записи по источнику
        tp_rows = group[tp_mask]
        vio_rows = group[vio_mask]
        other_rows = group[~tp_mask & ~vio_mask]

        if not tp_rows.empty:
            # Если есть записи из 'ТП', берем ВСЕ из них
            result_rows.append(tp_rows)
        elif not vio_rows.empty:
            # Если записей из 'ТП' нет, берем ПЕРВУЮ из 'ВиО'
            result_rows.append(vio_rows.head(1))
        elif not other_rows.empty:
            # Если есть другие источники, берем первую запись
            result_rows.append(other_rows.head(1))

    # Собираем результат
    if result_rows:
        result = pd.concat(result_rows, ignore_index=True)
    else:
        result = pd.DataFrame(columns=df.columns)

    # Удаляем временные столбцы
    result = result.drop(columns=["_q_norm", "_source_norm"])

    return result


def split_by_source(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Разделяет DataFrame на ТП и ВиО"""
    if df.empty:
        return None, None

    # Нормализуем source
    source_norm = df["source"].astype("string").str.strip().str.casefold()

    tp_mask = source_norm == "тп"
    vio_mask = source_norm == "вио"

    tp_df = df[tp_mask].copy() if tp_mask.any() else None
    vio_df = df[vio_mask].copy() if vio_mask.any() else None

    return tp_df, vio_df
