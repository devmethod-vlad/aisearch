import io
from datetime import UTC, datetime

import pandas as pd
from openpyxl.worksheet.worksheet import Worksheet


def build_export_dataframe(
    df: pd.DataFrame,
    field_mapping: dict[str, str],
    *,
    excluded_columns: set[str] | None = None,
) -> pd.DataFrame:
    """Готовит датафрейм для вкладки "Знания".

    Функция используется updater-пайплайном после дедупликации знаний.
    Она выполняет обратное переименование колонок: из внутренних имен
    проекта (`source`, `ext_id`, `question` и т.д.) в исходные русские
    названия из field_mapping.json.

    Порядок колонок сохраняется по порядку ключей JSON-маппинга.
    Колонки, которых нет в маппинге, добавляются в конец без переименования.
    """
    reversed_mapping = {internal_name: russian_name for russian_name, internal_name in field_mapping.items()}

    renamed_df = df.rename(columns=reversed_mapping).copy()
    if excluded_columns:
        renamed_df = renamed_df.drop(
            columns=[column for column in excluded_columns if column in renamed_df.columns]
        )

    mapped_columns_order = [column_name for column_name in field_mapping if column_name in renamed_df.columns]
    extra_columns = [column_name for column_name in renamed_df.columns if column_name not in mapped_columns_order]

    final_columns = mapped_columns_order + extra_columns
    return renamed_df[final_columns]


def build_source_statistics(
    df: pd.DataFrame,
    *,
    source_column: str = "Источник",
    id_column: str = "ID",
    count_column: str = "Количество уникальных ID",
) -> pd.DataFrame:
    """Строит статистику уникальных знаний по источникам.

    Функция используется при формировании Excel-отчета для Confluence.
    Она ожидает датафрейм с исходными русскими названиями колонок и
    считает количество уникальных значений в колонке "ID" для каждого
    значения из колонки "Источник".

    Возвращает датафрейм с колонками:
    - "Источник"
    - "Количество уникальных ID"
    """
    missing_columns = [column for column in (source_column, id_column) if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Для формирования статистики отсутствуют колонки: "
            f"{missing_columns}"
        )

    stats_df = (
        df.groupby(source_column, dropna=False)[id_column]
        .nunique(dropna=True)
        .reset_index(name=count_column)
    )
    if source_column != "Источник":
        stats_df = stats_df.rename(columns={source_column: "Источник"})

    return stats_df


def build_filter_comparison_statistics(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    *,
    source_column: str = "source",
    id_column: str = "ext_id",
    before_count_column: str = "Всего (до фильтрации)",
    after_count_column: str = "Всего (после фильтрации)",
) -> pd.DataFrame:
    before_stats = build_source_statistics(
        before_df,
        source_column=source_column,
        id_column=id_column,
        count_column=before_count_column,
    )
    after_stats = build_source_statistics(
        after_df,
        source_column=source_column,
        id_column=id_column,
        count_column=after_count_column,
    )

    merged_df = before_stats.merge(after_stats, on="Источник", how="outer")
    merged_df[[before_count_column, after_count_column]] = (
        merged_df[[before_count_column, after_count_column]].fillna(0).astype(int)
    )
    return merged_df[["Источник", before_count_column, after_count_column]]


def _apply_excel_formatting(
    source_df: pd.DataFrame,
    worksheet: Worksheet,
) -> None:
    """Применяет базовое форматирование листа Excel для читаемости.

    Функция используется при формировании Excel-отчета с дедуплицированными
    знаниями. Она фиксирует строку заголовков, включает автофильтр и подбирает
    ширину колонок по максимальной длине заголовка или значения.

    Значения `pd.NA`, `NaN` и `None` приводятся к пустой строке, чтобы расчет
    длины ячейки не падал на пропущенных данных.
    """
    worksheet.freeze_panes = "A2"

    if not source_df.empty:
        worksheet.auto_filter.ref = worksheet.dimensions

    for index, column_name in enumerate(source_df.columns, start=1):
        column_values = source_df[column_name].fillna("").astype(str)
        max_cell_length = (
            column_values.map(len).max() if not column_values.empty else 0
        )

        header_length = len(str(column_name))
        width = min(max(header_length, int(max_cell_length)) + 2, 80)

        column_letter = worksheet.cell(row=1, column=index).column_letter
        worksheet.column_dimensions[column_letter].width = width


def build_deduplicated_knowledge_excel(
    df: pd.DataFrame,
    field_mapping: dict[str, str],
    *,
    statistics_df: pd.DataFrame | None = None,
    excluded_columns: set[str] | None = None,
) -> bytes:
    """Формирует Excel-файл с дедуплицированными знаниями.

    Функция используется `UpdaterService.update_all` после шага
    `dedup_by_question_any`.

    В Excel создаются две вкладки:
    - "Знания": дедуплицированные знания с исходными русскими названиями колонок;
    - "Статистика": количество уникальных ID по каждому источнику.

    Возвращает содержимое `.xlsx` файла в виде bytes, чтобы файл можно было
    загрузить в Confluence без сохранения на диск.
    """
    knowledge_df = build_export_dataframe(
        df=df,
        field_mapping=field_mapping,
        excluded_columns=excluded_columns,
    )
    final_statistics_df = (
        statistics_df if statistics_df is not None else build_source_statistics(knowledge_df)
    )

    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        knowledge_df.to_excel(writer, index=False, sheet_name="Знания")
        final_statistics_df.to_excel(writer, index=False, sheet_name="Статистика")

        knowledge_ws = writer.sheets["Знания"]
        stats_ws = writer.sheets["Статистика"]

        _apply_excel_formatting(knowledge_df, knowledge_ws)
        _apply_excel_formatting(final_statistics_df, stats_ws)

    return buffer.getvalue()


def build_export_file_name(template: str, timestamp: datetime | None = None) -> str:
    """Формирует имя Excel-файла для загрузки в Confluence.

    Используется настройка `EXTRACT_DEDUPLICATED_EXCEL_FILE_NAME_TEMPLATE`.
    Если шаблон содержит `{timestamp}`, подставляет текущий timestamp.
    Если расширение `.xlsx` не указано, добавляет его автоматически.
    """
    ts = timestamp or datetime.now(tz=UTC)
    file_name = template.format(timestamp=ts.strftime("%Y%m%d_%H%M%S"))

    if not file_name.lower().endswith(".xlsx"):
        file_name = f"{file_name}.xlsx"

    return file_name
