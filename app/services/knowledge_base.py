import os
import re
from multiprocessing.pool import AsyncResult

import pandas as pd
import shortuuid

from app.api.v1.dto.responses.knowledge_base import CollectDataInfoResponse, CollectDataResponse
from app.api.v1.dto.responses.taskmanager import TaskQueryResponse
from app.common.logger import AISearchLogger
from app.domain.adapters.interfaces import IConfluenceAdapter, IEduAdapter, IGoogleTablesAdapter
from app.domain.exceptions import ConfluenceException, GoogleTablesException
from app.domain.schemas.knowledge_base import KnowledgeError, KnowledgeErrorType
from app.infrastructure.utils.universal import (
    convert_dataframe_to_list_of_tuples,
    convert_dataframe_to_xlsx_file,
)
from app.infrastructure.worker.tasks.knowledge_base import (
    task_collect_all_data_from_confluence,
    task_collect_data_from_confluence_by_page_id_task,
)
from app.services.interfaces import IKnowledgeBaseService
from app.settings.config import AppSettings


class KnowledgeBaseService(IKnowledgeBaseService):
    """Сервис базы знаний"""

    def __init__(
        self,
        confluence_adapter: IConfluenceAdapter,
        google_tables_adapter: IGoogleTablesAdapter,
        edu_adapter: IEduAdapter,
        app_config: AppSettings,
        logger: AISearchLogger,
    ):
        self.confluence_adapter = confluence_adapter
        self.google_tables_adapter = google_tables_adapter
        self.edu_adapter = edu_adapter
        self.separator = app_config.knowledge_base_roles_separator
        self.logger = logger

    async def collect_data_from_confluence_by_page_id(self, page_id: str) -> CollectDataResponse:
        """Сбор данных БЗ из страницы Confluence"""
        knowledge_df, knowledge_errors, collect_info = (
            await self._collect_knowledge_base_and_get_errors(page_id=page_id)
        )
        if knowledge_df.empty:
            return CollectDataResponse(
                parsing_error=bool(knowledge_errors),
                message="Данные отсутствуют",
                pages_info=[collect_info],
            )

        try:
            google_df = await self.google_tables_adapter.load_google_sheet_to_dataframe(
                url=self.google_tables_adapter.megatable_url
            )
            enriched_df = self._enrich_data(knowledge_df=knowledge_df, google_df=google_df)
        except GoogleTablesException as e:
            error_text = f"Не удалось обогатить данные БЗ ({str(e)})"
            self.logger.warning(error_text)
            return CollectDataResponse(
                parsing_error=bool(knowledge_errors), message=error_text, pages_info=[collect_info]
            )

        sorted_df = self._sort_df_by_row_id(df=enriched_df)
        sorted_df = self._sort_knowledge_df_columns(knowledge_df=sorted_df)
        knowledge_xlsx = convert_dataframe_to_xlsx_file(df=sorted_df)
        await self.edu_adapter.create_or_update_file_on_edu(
            filename=f"KB_wiki_{page_id}.xlsx", file_data=knowledge_xlsx
        )
        # self._save_enriched_df_to_excel(
        #     df=sorted_df, path_filename="volumes/KB_wiki.xlsx", overwrite=True
        # )

        if bool(knowledge_errors):
            errors_df = KnowledgeError.to_dataframe(knowledge_errors)
            errors_df = self._sort_df_by_page_id_from_url(
                df=errors_df, col_name="URL на wiki-страницу"
            )
            errors_df = self._sort_errors_df_columns(errors_df=errors_df)
            errors_xlsx = convert_dataframe_to_xlsx_file(df=errors_df)
            await self.edu_adapter.create_or_update_file_on_edu(
                filename=f"ERRORS_{page_id}.xlsx", file_data=errors_xlsx
            )
            # self._save_errors_df_to_excel(errors_df, path_filename="volumes/ERRORS.xlsx")

        return CollectDataResponse(
            parsing_error=bool(knowledge_errors),
            message="Данные собраны",
            pages_info=[collect_info],
        )

    async def collect_data_from_confluence_by_page_id_detached(
        self, page_id: str
    ) -> TaskQueryResponse:
        """Сбор данных БЗ из страницы Confluence в фоновом режиме"""
        result_key = shortuuid.uuid()
        task: AsyncResult = task_collect_data_from_confluence_by_page_id_task.apply_async(
            args=[page_id],
            task_id=f"knowledge-base:{result_key}",
        )

        return TaskQueryResponse(
            task_id=task.task_id,
            task_status_url=f"/taskmanager/{task.task_id}/{result_key}",
        )

    async def collect_all_data_from_confluence(self) -> CollectDataResponse:
        """Сбор всех данных БЗ"""
        try:
            df_minitable: pd.DataFrame = (
                await self.google_tables_adapter.load_google_sheet_to_dataframe(
                    url=self.google_tables_adapter.minitable_url
                )
            )
        except GoogleTablesException as e:
            error_text = f"Не удалось собрать данные из минитаблицы ({str(e)})"
            self.logger.warning(error_text)
            return CollectDataResponse(message=error_text)

        pages_ids = (
            df_minitable.rename(columns=df_minitable.iloc[0])
            .drop(df_minitable.index[0])["pageId"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        knowledge_df_extended = pd.DataFrame()
        knowledge_errors: list[KnowledgeError] = []
        collection_info: list[CollectDataInfoResponse] = []

        for page_id in pages_ids:
            knowledge_df, errors, info = await self._collect_knowledge_base_and_get_errors(
                page_id=page_id
            )
            knowledge_df_extended = pd.concat([knowledge_df_extended, knowledge_df])
            knowledge_errors += errors
            collection_info.append(info)

        if knowledge_df_extended.empty:
            return CollectDataResponse(
                parsing_error=bool(knowledge_errors),
                message="Данные отсутствуют",
                pages_info=collection_info,
            )

        try:
            google_df = await self.google_tables_adapter.load_google_sheet_to_dataframe(
                url=self.google_tables_adapter.megatable_url
            )
            enriched_df = self._enrich_data(knowledge_df=knowledge_df_extended, google_df=google_df)
        except GoogleTablesException as e:
            error_text = f"Не удалось обогатить данные БЗ ({str(e)})"
            self.logger.warning(error_text)
            return CollectDataResponse(parsing_error=bool(knowledge_errors), message=error_text)

        sorted_df = self._sort_df_by_row_id(df=enriched_df)
        sorted_df = self._sort_knowledge_df_columns(knowledge_df=sorted_df)
        knowledge_xlsx = convert_dataframe_to_xlsx_file(df=sorted_df)
        await self.edu_adapter.create_or_update_file_on_edu(
            filename="KB_wiki.xlsx", file_data=knowledge_xlsx
        )
        # self._save_enriched_df_to_excel(
        #     df=sorted_df, path_filename="volumes/KB_wiki.xlsx", overwrite=True
        # )

        if bool(knowledge_errors):
            errors_df = KnowledgeError.to_dataframe(knowledge_errors)
            errors_df = self._sort_df_by_page_id_from_url(
                df=errors_df, col_name="URL на wiki-страницу"
            )
            errors_df = self._sort_errors_df_columns(errors_df=errors_df)
            errors_xlsx = convert_dataframe_to_xlsx_file(df=errors_df)
            await self.edu_adapter.create_or_update_file_on_edu(
                filename="ERRORS.xlsx", file_data=errors_xlsx
            )
            # self._save_errors_df_to_excel(errors_df_extended, path_filename="volumes/ERRORS.xlsx")

        return CollectDataResponse(
            parsing_error=bool(knowledge_errors),
            message="Данные собраны",
            pages_info=collection_info,
        )

    async def collect_all_data_from_confluence_detached(self) -> CollectDataResponse:
        """Сбор всех данных БЗ в фоновом режиме"""
        task: AsyncResult = task_collect_all_data_from_confluence.apply_async(
            task_id=f"knowledge-base:{shortuuid.uuid()}",
        )

        return TaskQueryResponse(
            task_id=task.task_id,
            task_status_url=f"/taskmanager/{task.task_id}/{shortuuid.uuid()}",
        )

    async def _collect_knowledge_base_and_get_errors(
        self, page_id: str
    ) -> tuple[pd.DataFrame, list[KnowledgeError], CollectDataInfoResponse]:
        """Сбор данных БЗ и ошибок парсинга со страницы"""
        knowledge_df: pd.DataFrame = pd.DataFrame()
        knowledge_errors: list[KnowledgeError] = []

        try:
            html_content: str = await self.confluence_adapter.get_html_content_from_page(
                page_id=page_id
            )
            knowledge_df, errors = self.confluence_adapter.get_knowledge_base_from_html(
                html_content=html_content, page_id=page_id
            )
        except ConfluenceException as e:
            error_text = f"Не удалось собрать данные со страницы {page_id} ({str(e)})"
            self.logger.warning(error_text)
            if e.status_code == 404:
                knowledge_errors.append(
                    KnowledgeError(
                        url=f"{self.confluence_adapter.confluence_url}/pages/viewpage.action?pageId={page_id}",
                        error_type=KnowledgeErrorType.PAGE_NOT_FOUND,
                    )
                )
            return (
                knowledge_df,
                knowledge_errors,
                CollectDataInfoResponse(page_id=page_id, message=error_text),
            )

        knowledge_errors += errors

        if knowledge_df.empty:
            error_text = f"Данные БЗ отсутствуют на страницe {page_id}"
            self.logger.warning(error_text)
            return (
                knowledge_df,
                knowledge_errors,
                CollectDataInfoResponse(page_id=page_id, message=error_text),
            )

        knowledge_df, duplicates_errors = self._remove_duplicates_and_get_errors(
            knowledge_df=knowledge_df
        )
        knowledge_errors += duplicates_errors

        success_text = f"Данные со страницы {page_id} собраны"
        self.logger.info(success_text)
        return (
            knowledge_df,
            knowledge_errors,
            CollectDataInfoResponse(page_id=page_id, message=success_text),
        )

    def _remove_duplicates_and_get_errors(
        self, knowledge_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[KnowledgeError]]:
        """Удаление дубликатов и сбор ошибок"""
        errors: list[KnowledgeError] = []
        knowledge_df, knowledge_df_removed, df_original_copies = self._remove_row_id_duplicates(
            df=knowledge_df
        )

        if not knowledge_df_removed.empty:
            for page_id, row_id, content_from_error_desc in convert_dataframe_to_list_of_tuples(
                df=pd.concat([df_original_copies, knowledge_df_removed]),
                columns=["ID страницы", "ID", "Вопрос (clean)"],
            ):
                errors.append(
                    KnowledgeError(
                        url=f"{self.confluence_adapter.confluence_url}/pages/viewpage.action?pageId={page_id}",
                        error_type=KnowledgeErrorType.KNOWLEDGE_DUPLICATE,
                        knowledge_number=row_id,
                        column_name="Номер знания",
                        content_from_error_desc=content_from_error_desc,
                    )
                )
                self.logger.warning(f"({page_id} | {row_id}): Дубликат номера знания")

        return knowledge_df, errors

    def _remove_row_id_duplicates(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Удаление дубликатов номеров знаний"""
        if "ID" not in df.columns:
            raise ValueError("Колонка 'ID' отсутствует в DataFrame")

        duplicated_mask = df.duplicated(subset=["ID"], keep=False)
        duplicated_rows_mask = df.duplicated(subset=["ID"], keep="first")
        df_removed = df[duplicated_rows_mask].copy()
        df_result = df.drop_duplicates(subset=["ID"], keep="first").copy()
        df_original_copies = df[duplicated_mask & ~duplicated_rows_mask].copy()

        return df_result, df_removed, df_original_copies

    def _enrich_data(self, knowledge_df: pd.DataFrame, google_df: pd.DataFrame) -> pd.DataFrame:
        """Обогащение данных БЗ"""
        original_roles = self._get_roles_from_df(google_df, knowledge_df)

        knowledge_df = self.confluence_adapter.preprocess_knowledge_base_data(df=knowledge_df)
        google_df = self.google_tables_adapter.preprocess_megatable_data(df=google_df)

        enriched_df = self._merge_knowledge_base_with_google(
            knowledge_df=knowledge_df, google_df=google_df, separator=self.separator
        )
        enriched_df = self._restore_original_roles(
            df=enriched_df, roles_set=original_roles, separator=self.separator
        )

        return enriched_df

    def _get_roles_from_df(self, *dfs: pd.DataFrame) -> set[str]:
        """Получение ролей (колонка 'Роль') из DataFrame"""
        unique_roles = {}

        for df in dfs:
            if "Роль" not in df.columns:
                raise ValueError("Колонка 'Роль' отсутствует в одном из DataFrame'ов")

            for roles_in_cell in df["Роль"].dropna():
                if isinstance(roles_in_cell, str):
                    roles_list = [roles_in_cell]
                elif isinstance(roles_in_cell, list):
                    roles_list = roles_in_cell
                else:
                    continue

                for role in roles_list:
                    stripped_role = role.strip()
                    if stripped_role:
                        normalized_role = stripped_role.lower()
                        if normalized_role not in unique_roles:
                            unique_roles[normalized_role] = stripped_role

        return set(unique_roles.values())

    def _restore_original_roles(
        self, df: pd.DataFrame, roles_set: set[str], separator: str
    ) -> pd.DataFrame:
        """Восстановление оригинальных ролей"""
        if "Роль" not in df.columns:
            raise ValueError("Колонка 'Роль' отсутствует в DataFrame")

        normalized_to_original = {role.lower(): role for role in roles_set}

        def replace_role(role_string: str) -> str:
            roles_list = [role.strip() for role in role_string.split(separator) if role.strip()]
            updated_roles = [normalized_to_original.get(role.lower(), role) for role in roles_list]
            return separator.join(updated_roles)

        if "Роль" in df.columns:
            df["Роль"] = df["Роль"].apply(replace_role)

        return df

    def _merge_knowledge_base_with_google(
        self, knowledge_df: pd.DataFrame, google_df: pd.DataFrame, separator: str = ";"
    ) -> pd.DataFrame:
        """Слияние фрейма БЗ с фреймом Google"""
        not_empty_role_mask = knowledge_df["Роль"].str.strip() != ""
        knowledge_df_with_roles = knowledge_df.loc[not_empty_role_mask]

        temp_df = pd.merge(
            knowledge_df_with_roles[["ID страницы", "ID", "Роль"]],
            google_df[["ID страницы из БЗ", "Роль", "Проект / Пространство"]],
            left_on=["ID страницы", "Роль"],
            right_on=["ID страницы из БЗ", "Роль"],
            how="left",
        )
        temp_df = (
            temp_df[["ID страницы", "ID", "Проект / Пространство"]]
            .dropna()
            .drop_duplicates(subset=["ID страницы", "ID", "Проект / Пространство"])
            .sort_values(["ID страницы", "ID"])
            .reset_index(drop=True)
        )
        knowledge_df = pd.merge(
            knowledge_df,
            temp_df[["ID страницы", "ID", "Проект / Пространство"]],
            on=["ID страницы", "ID"],
            how="left",
        ).rename(columns={"Проект / Пространство": "Пространство"})
        knowledge_df = knowledge_df.groupby(["ID страницы", "ID"], as_index=False).agg(
            {
                "Роль": lambda x: (
                    separator.join({role.strip() for role in x.dropna() if role.strip()})
                    if any(role.strip() for role in x.dropna())
                    else ""
                ),
                "Пространство": lambda x: (
                    separator.join({space.strip() for space in x.dropna() if space.strip()})
                    if any(space.strip() for space in x.dropna())
                    else ""
                ),
                **dict.fromkeys(set(knowledge_df.columns) - {"Роль", "Пространство"}, "first"),
            }
        )
        google_df = google_df.groupby(["ID страницы из БЗ"], as_index=False).agg(
            {
                "Роль": lambda x: separator.join(
                    {role.strip() for role in x.dropna() if role.strip()}
                    if any(role.strip() for role in x.dropna())
                    else ""
                ),
                "Проект / Пространство": lambda x: (
                    separator.join({space.strip() for space in x.dropna() if space.strip()})
                    if any(space.strip() for space in x.dropna())
                    else ""
                ),
                "Продукт / Сервис": lambda x: (
                    separator.join({product.strip() for product in x.dropna() if product.strip()})
                    if any(product.strip() for product in x.dropna())
                    else ""
                ),
            }
        )

        empty_space_mask = knowledge_df["Пространство"].str.strip() == ""
        knowledge_df.loc[empty_space_mask, "Пространство"] = knowledge_df.loc[
            empty_space_mask, "ID страницы"
        ].map(
            google_df.drop_duplicates("ID страницы из БЗ").set_index("ID страницы из БЗ")[
                "Проект / Пространство"
            ]
        )

        empty_role_mask = knowledge_df["Роль"].str.strip() == ""
        knowledge_df.loc[empty_role_mask, "Роль"] = knowledge_df.loc[
            empty_role_mask, "ID страницы"
        ].map(google_df.drop_duplicates("ID страницы из БЗ").set_index("ID страницы из БЗ")["Роль"])

        knowledge_df["Продукт"] = knowledge_df["ID страницы"].map(
            google_df.drop_duplicates("ID страницы из БЗ").set_index("ID страницы из БЗ")[
                "Продукт / Сервис"
            ]
        )

        return knowledge_df.fillna("")

    def _save_enriched_df_to_excel(
        self, df: pd.DataFrame, path_filename: str, overwrite: bool = False
    ) -> None:
        """Сохранение обогащённого фрейма БЗ в excel-файл"""
        if overwrite or not os.path.exists(path_filename):
            updated_df = df
        else:
            existing_df = pd.read_excel(path_filename, engine="openpyxl")
            page_ids_to_update = df["ID страницы"].unique()
            existing_df = existing_df[~existing_df["ID страницы"].isin(page_ids_to_update)]
            updated_df = pd.concat([existing_df, df], ignore_index=True)
            updated_df = updated_df.drop_duplicates(keep="last")

        # updated_df = updated_df.sort_values(by="ID страницы", ascending=True)
        updated_df.to_excel(path_filename, index=False, engine="openpyxl")

    def _sort_df_by_row_id(self, df: pd.DataFrame, col_name: str = "ID") -> pd.DataFrame:
        """Сортировка DataFrame по номеру знания"""
        df[["letters", "numbers"]] = df[col_name].str.split("-", expand=True)
        df["numbers"] = df["numbers"].astype(int)
        df_sorted = df.sort_values(["letters", "numbers"])
        return df_sorted.drop(["letters", "numbers"], axis=1).reset_index(drop=True)

    def _get_page_id_from_url(self, url: str) -> int:
        """Получение id страницы из URL"""
        match = re.search(r"pageId=(\d+)", url)
        return int(match.group(1)) if match else 0

    def _sort_df_by_page_id_from_url(
        self, df: pd.DataFrame, col_name: str = "URL на wiki-страницу"
    ) -> pd.DataFrame:
        """Сортировка DataFrame по id страницы из ссылки"""
        if df.empty or col_name not in df.columns:
            return df.copy()

        return df.sort_values(
            by=col_name, key=lambda x: x.map(self._get_page_id_from_url)
        ).reset_index(drop=True)

    def _sort_knowledge_df_columns(self, knowledge_df: pd.DataFrame) -> pd.DataFrame:
        """Упорядочивание колонок DataFrame БЗ"""
        return knowledge_df[
            [
                "Источник",
                "ID",
                "ID страницы",
                "Актуально",
                "2 линия",
                "Роль",
                "Компонент",
                "Вопрос (markdown)",
                "Вопрос (clean)",
                "Анализ ошибки (markdown)",
                "Анализ ошибки (clean)",
                "Ответ (markdown)",
                "Ответ (clean)",
                "Для пользователя",
                "Jira",
            ]
        ]

    def _sort_errors_df_columns(self, errors_df: pd.DataFrame) -> pd.DataFrame:
        """Упорядочивание колонок DataFrame ошибок"""
        return errors_df[
            [
                "URL на wiki-страницу",
                "Номер знания",
                "Тип ошибки",
                "Название столбца с ошибкой",
                'Содержимое из "Описание ошибки"',
            ]
        ]

    def _save_errors_df_to_excel(self, errors_df: pd.DataFrame, path_filename: str) -> None:
        """Сохранение файла ошибок в excel-файл"""
        errors_df.to_excel(path_filename, index=False, engine="openpyxl")
