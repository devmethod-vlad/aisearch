import logging
from typing import Any

import numpy as np
import pandas as pd

def rename_dataframe(
    df: pd.DataFrame, field_mapping: dict[str, str] | None = None
)->pd.DataFrame:
    local_field_mapping = {
        "Источник": "source",
        "ID": "ext_id",
        "ID страницы": "page_id",
        "Актуально": "actual",
        "2 линия": "second_line",
        "Роль": "role",
        "Продукт": "product",
        "Пространство": "space",
        "Компонент": "component",
        "Вопрос (markdown)": "question_md",
        "Вопрос (clean)": "question",
        "Анализ ошибки (markdown)": "analysis_md",
        "Анализ ошибки (clean)": "analysis",
        "Ответ (markdown)": "answer_md",
        "Ответ (clean)": "answer",
        "Для пользователя": "for_user",
        "Jira": "jira",
        "Обновлено": "modified_at",
    }
    if not field_mapping:
        field_mapping = local_field_mapping

    df.rename(columns=field_mapping, inplace=True)
    return df

def prepare_dataframe(
    df: pd.DataFrame,
    logger: logging.Logger
) -> tuple[Any, list[dict], pd.DataFrame]:
    """Унифицированная очистка/фильтрация данных для pre_launch и updater."""
    # cols = df.columns.difference(["page_id"])
    # df[cols] = df[cols].astype("string").fillna("")
    df = df.astype("string").fillna("")

    df["ext_id"] = df["ext_id"].astype(str).str.strip()
    df["source"] = df["source"].astype(str).str.strip()

    initial_count = len(df)
    logger.info(f"Получено строк: {initial_count}")

    # базовая фильтрация
    base_mask = df["question"].notna() & (df["question"].astype(str).str.len() > 0)
    df = df[base_mask]
    logger.info(f"После базовых фильтров: {len(df)}")

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

    logger.info(f"Итоговое количество строк: {len(df_final)}")

    documents = df_final["question"].astype(str).tolist()
    metadata = df_final.to_dict(orient="records")

    return documents, metadata, df_final


def dedup_by_question_any(
    df: pd.DataFrame,
    target_source: str = "ТП",
    question_col: str = "question",
    source_col: str = "source",
) -> pd.DataFrame:
    df = df.copy()

    # нормализуем столбец для сравнения
    q_norm = df[question_col].astype("string").str.strip().str.casefold()

    # чтобы NaN в столбец не схлопывались в один "дубликат" (с fillna('') неактуально, но пусть будет)
    na = q_norm.isna()
    if na.any():
        q_norm = q_norm.where(~na, "__NA__" + df.index[na].astype(str))

    # нормализуем поле источника
    target_norm = str(target_source).strip().casefold()
    # условие для выбора источника
    is_target = (
        df[source_col].astype("string").str.strip().str.casefold().eq(target_norm)
    )
    # условие для фильтра по источнику
    priority = np.where(is_target, 0, 1)

    tmp = df.assign(_q_norm=q_norm, _priority=priority).sort_values(
        ["_q_norm", "_priority"], kind="mergesort"
    )

    tmp = tmp.drop_duplicates("_q_norm", keep="first")

    return tmp.drop(columns=["_q_norm", "_priority"])
