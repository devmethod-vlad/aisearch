import re
import typing as tp

import aiohttp
import pandas as pd
from bs4 import BeautifulSoup, Tag

from app.common.logger import AISearchLogger
from app.domain.adapters.interfaces import IConfluenceAdapter
from app.domain.exceptions import ConfluenceException
from app.domain.schemas.knowledge_base import KnowledgeError, KnowledgeErrorType
from app.infrastructure.utils.parse.extract import (
    CombinedRowsData,
    extract_clean_text,
    extract_markdown_text,
    find_header_for_cell,
    get_combined_rows,
)
from app.infrastructure.utils.universal import (
    cast_to_int,
    generate_row_order_mapping,
    normalize_string,
)
from app.settings.config import AppSettings


class ConfluenceAdapter(IConfluenceAdapter):
    """Адаптер Confluence"""

    def __init__(self, app_settings: AppSettings, logger: AISearchLogger):
        self.__confluence_url = app_settings.confluence_url
        self.token = app_settings.confluence_token
        self.main_header_text = app_settings.knowledge_base_main_header_text
        self.target_columns = app_settings.knowledge_base_target_columns
        self.roles_separator = app_settings.knowledge_base_roles_separator
        self.logger = logger

    @property
    def confluence_url(self) -> str:
        """Confluence URL"""
        return self.__confluence_url

    async def get_html_content_from_page(self, page_id: str) -> str:
        """Получение HTML из страницы Confluence"""
        url = f"{self.confluence_url}/rest/api/content/{page_id}?expand=body.view"
        headers = {"Authorization": f"Bearer {self.token}"}

        async with aiohttp.ClientSession() as session, session.get(
            url, headers=headers
        ) as response:
            if response.status == 200:
                html_content = (await response.json())["body"]["view"]["value"]
                # with open(f"tests/mocks/knowledges/{page_id}.html", "w") as file:
                #     file.write(html_content)
                return html_content
            else:
                raise ConfluenceException(
                    "Ошибка при получении страницы", status_code=response.status
                )

    def get_knowledge_base_from_html(
        self,
        html_content: str,
        page_id: str,
        filter_actual: bool = False,
        filter_for_user: bool = False,
    ) -> tuple[pd.DataFrame, list[KnowledgeError]]:
        """Получение контента БЗ из HTML"""
        page_url = f"{self.confluence_url}/pages/viewpage.action?pageId={page_id}"
        errors: list[KnowledgeError] = []
        soup = BeautifulSoup(re.sub(r">\s+<", "> <", html_content), "html.parser")
        for h1 in soup.find_all("h1"):
            full_text = h1.get_text(strip=True)
            if self.main_header_text.lower() in full_text.lower():
                knowledge_base_header = h1
                break
        else:
            errors.append(
                KnowledgeError(
                    url=page_url,
                    error_type=KnowledgeErrorType.MAIN_HEADER_MISSING,
                )
            )
            self.logger.warning(f"({page_id}): Заголовок {self.main_header_text} не найден")
            return pd.DataFrame(), errors

        tables: list[Tag] = []
        for sibling in knowledge_base_header.find_all_next():
            if sibling.name == "table":
                tables.append(sibling)

        if len(tables) == 0:
            errors.append(
                KnowledgeError(
                    url=page_url,
                    error_type=KnowledgeErrorType.DATA_MISSING,
                )
            )
            self.logger.warning(f"({page_id}): Не было найдено ни одной актуальной таблицы")
            return pd.DataFrame(), errors

        table = tables[0]

        excluded_tables = {table}
        nested_tables = table.find_all("table")
        excluded_tables.update(nested_tables)
        next_tables = [table for table in tables if table not in excluded_tables]

        # next_tables: list[Tag] = []
        # for sibling in table.find_next_siblings():
        #     if len(next_tables) > 0:
        #         break
        #     if sibling.name == "table":
        #         tables.append(sibling)
        #     else:
        #         next_tables.extend(sibling.find_all("table", limit=1))

        if len(next_tables) > 0:
            errors.append(
                KnowledgeError(
                    url=page_url,
                    error_type=KnowledgeErrorType.SEVERAL_TABLES,
                )
            )
            self.logger.warning(f"({page_id}): Было найдено более 1 таблицы")

        filtered_rows: list[dict] = []

        # try:
        header_row = table.find("tr")
        if not header_row.find("th"):
            errors.append(
                KnowledgeError(
                    url=page_url,
                    error_type=KnowledgeErrorType.HEADER_ROW_MISSING,
                )
            )
            self.logger.warning(f"({page_id}): Отсутствует строка заголовков")
            return pd.DataFrame(), errors

        headers = [header.get_text(strip=True) for header in header_row.find_all("th")]
        if len(headers) != len(self.target_columns):
            errors.append(
                KnowledgeError(
                    url=page_url,
                    error_type=KnowledgeErrorType.NUMBER_OF_COLUMNS,
                )
            )
            self.logger.warning(
                f"({page_id}): Несоответствие количества столбцов. "
                f"Ожидалось: {len(self.target_columns)}, получено: {len(headers)}"
            )
            return pd.DataFrame(), errors

        if sorted([normalize_string(s).lower() for s in headers]) != sorted(
            [normalize_string(s).lower() for s in self.target_columns]
        ):
            errors.append(
                KnowledgeError(
                    url=page_url,
                    error_type=KnowledgeErrorType.COLUMN_NAMES,
                )
            )
            self.logger.warning(
                f"({page_id}): Несоответствие названий столбцов. "
                f"Ожидалось: {self.target_columns}, получено: {headers}"
            )
            return pd.DataFrame(), errors

        all_td = table.find_all("tr")
        rows = [td for td in all_td if td.find_parent("table") == table]

        cells_ordering: list[int] = generate_row_order_mapping(
            original=normalize_string(";".join(self.target_columns)).lower(),
            target=normalize_string(rows[0].get_text(separator=";", strip=True)).lower(),
        )
        combined_rows_data: list[CombinedRowsData] = get_combined_rows(table_tag=table)

        amount_of_rows_to_skip = 0

        for row in rows[1:]:

            if amount_of_rows_to_skip:
                amount_of_rows_to_skip -= 1
                continue

            cells = row.find_all(["td"])
            row_elements: dict[str, Tag] = {
                self.target_columns[i]: cells[j] for i, j in zip(range(len(cells)), cells_ordering)
            }
            row_id = row_elements["Номер знания"].get_text(strip=True).upper()

            if not row_id:
                errors.append(
                    KnowledgeError(
                        url=page_url,
                        knowledge_number=row_id,
                        error_type=KnowledgeErrorType.KNOWLEDGE_NUMBER_NOT_SPECIFIED,
                        column_name="Номер знания",
                        content_from_error_desc=extract_clean_text(row_elements["Описание ошибки"]),
                    )
                )
                self.logger.warning(f"({page_id}): Отсутствует номер знания")
                continue

            pass_row = False
            for combined_rows_info in combined_rows_data:
                if row in combined_rows_info.rows:
                    cols = ", ".join(combined_rows_info.column_names)
                    unique_combined_row_ids = set()

                    for combined_row in combined_rows_info.rows:
                        combined_row_cells = combined_row.find_all(["td"])
                        combined_row_elements: dict[str, Tag] = {
                            self.target_columns[i]: combined_row_cells[j]
                            for i, j in zip(range(len(combined_row_cells)), cells_ordering)
                        }
                        combined_row_id = (
                            combined_row_elements["Номер знания"].get_text(strip=True).upper()
                        )
                        if combined_row_id:
                            unique_combined_row_ids.add(combined_row_id)
                        else:
                            amount_of_rows_to_skip += 1

                    combined_row_ids = sorted(unique_combined_row_ids)
                    row_ids = (
                        ", ".join(combined_row_ids)
                        if len(combined_row_ids) > 1
                        else combined_row_ids[0]
                    )
                    kn_err = KnowledgeError(
                        url=page_url,
                        knowledge_number=row_ids,
                        error_type=KnowledgeErrorType.MERGED_ROWS,
                        column_name=cols,
                    )
                    if kn_err not in errors:
                        errors.append(kn_err)
                        self.logger.warning(
                            f"({page_id} | {row_ids}): " f"Объединённые ячейки в строках ({cols})"
                        )
                    pass_row = True
                    break
            if pass_row:
                continue

            pass_row = False
            for cell in cells:
                nested_tables = cell.find_all("table")
                if len(nested_tables) > 0:
                    col = find_header_for_cell(cell_tag=cell).get_text(strip=True)
                    errors.append(
                        KnowledgeError(
                            url=page_url,
                            knowledge_number=row_id,
                            error_type=KnowledgeErrorType.NESTED_TABLE,
                            column_name=col,
                            content_from_error_desc=extract_clean_text(
                                row_elements["Описание ошибки"]
                            ),
                        )
                    )
                    self.logger.warning(f"({page_id} | {row_id}): Вложенная таблица ({col})")
                    pass_row = True
            if pass_row:
                continue

            if not bool(re.fullmatch(r"^[A-ZА-ЯЁ]+-[0-9]+$", row_id)):
                errors.append(
                    KnowledgeError(
                        url=page_url,
                        knowledge_number=row_id,
                        error_type=KnowledgeErrorType.KNOWLEDGE_NUMBER_INCORRECT_FORMAT,
                        column_name="Номер знания",
                        content_from_error_desc=extract_clean_text(row_elements["Описание ошибки"]),
                    )
                )
                self.logger.warning(
                    f"({page_id} | {row_id}): Номер знания не соответствует формату"
                )
                continue

            if len(cells) != len(self.target_columns):
                errors.append(
                    KnowledgeError(
                        url=page_url,
                        knowledge_number=row_id,
                        error_type=KnowledgeErrorType.MERGED_COLUMNS,
                        content_from_error_desc=extract_clean_text(row_elements["Описание ошибки"]),
                    )
                )
                self.logger.warning(f"({page_id} | {row_id}): Объединённые ячейки в столбцах")
                continue

            for cell, col_name in [
                (row_elements[col_name], col_name)
                for col_name in ["Описание ошибки", "Анализ ошибки", "Шаблон ответа"]
            ]:
                unknown_attachments = cell.find_all(
                    lambda tag: (
                        tag.get("class")
                        and any("unknown-attachment" in cls for cls in tag.get("class"))
                    )
                    or (
                        tag.get("src")
                        and "/confluence/placeholder/unknown-attachment" in tag.get("src")
                    )
                )
                if unknown_attachments:
                    errors.append(
                        KnowledgeError(
                            url=page_url,
                            knowledge_number=row_id,
                            error_type=KnowledgeErrorType.UNKNOWN_ATTACHMENT,
                            column_name=col_name,
                            content_from_error_desc=extract_clean_text(
                                row_elements["Описание ошибки"]
                            ),
                        )
                    )
                    self.logger.warning(
                        f"({page_id} | {row_id}): Неизвестное вложение ({col_name})"
                    )
                    for element in unknown_attachments:
                        element.extract()

            if filter_actual and row_elements["Актуально"].get_text(strip=True).lower() == "нет":
                continue

            if (
                filter_for_user
                and row_elements["Для пользователя"].get_text(strip=True).lower() != "да"
            ):
                continue

            # БЗ ИП
            data_to_append: dict[str, tp.Any] = {}
            data_to_append["Источник"] = "БЗ"
            data_to_append["ID"] = row_id
            data_to_append["ID страницы"] = page_id
            data_to_append["Актуально"] = (
                row_elements["Актуально"].get_text(strip=True).capitalize()
            )
            data_to_append["2 линия"] = row_elements["2 линия"].get_text(strip=True)
            data_to_append["Роль"] = [
                normalize_string(role_name)
                for role_name in row_elements["Роль"]
                .get_text(strip=True)
                .split(self.roles_separator)
                if role_name
            ]
            data_to_append["Компонент"] = row_elements["Компонент"].get_text(strip=True)

            data_to_append["Вопрос (markdown)"] = extract_markdown_text(
                tag=row_elements["Описание ошибки"], confluence_url=self.confluence_url
            )
            data_to_append["Вопрос (clean)"] = extract_clean_text(
                tag=row_elements["Описание ошибки"]
            )

            data_to_append["Анализ ошибки (markdown)"] = extract_markdown_text(
                tag=row_elements["Анализ ошибки"], confluence_url=self.confluence_url
            )
            data_to_append["Анализ ошибки (clean)"] = extract_clean_text(
                tag=row_elements["Анализ ошибки"]
            )

            data_to_append["Ответ (markdown)"] = extract_markdown_text(
                tag=row_elements["Шаблон ответа"], confluence_url=self.confluence_url
            )
            data_to_append["Ответ (clean)"] = extract_clean_text(tag=row_elements["Шаблон ответа"])

            data_to_append["Для пользователя"] = (
                row_elements["Для пользователя"].get_text(strip=True).capitalize()
            )
            data_to_append["Jira"] = extract_markdown_text(
                tag=row_elements["Jira"], confluence_url=self.confluence_url
            )

            filtered_rows.append(data_to_append)

        if not filtered_rows:
            errors.append(
                KnowledgeError(
                    url=page_url,
                    error_type=KnowledgeErrorType.DATA_MISSING,
                )
            )
            self.logger.warning(f"({page_id}): На странице есть таблица, но нет актуальных строк")

        result_df = pd.DataFrame(filtered_rows)
        return result_df, errors

    def preprocess_knowledge_base_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Предобработка данных БЗ"""
        df.columns = df.columns.str.strip()
        df = df.explode("Роль")
        df["Роль"] = df["Роль"].apply(lambda role: normalize_string(role).lower())
        df["Роль"] = df["Роль"].fillna("").astype(str)
        df["ID страницы"] = df["ID страницы"].apply(cast_to_int).astype(int)
        return df
