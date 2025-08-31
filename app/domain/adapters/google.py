from io import BytesIO

import aiohttp
import pandas as pd

from app.common.logger import AISearchLogger
from app.domain.adapters.interfaces import IGoogleTablesAdapter
from app.domain.exceptions import GoogleTablesException
from app.infrastructure.utils.universal import cast_to_int, normalize_string
from app.settings.config import AppSettings


class GoogleTablesAdapter(IGoogleTablesAdapter):
    """Адаптер Google-таблиц"""

    def __init__(self, app_settings: AppSettings, logger: AISearchLogger):
        self.__minitable_url = app_settings.knowledge_base_minitable_google_link
        self.__megatable_url = app_settings.knowledge_base_megatable_google_link
        self.logger = logger

    @property
    def minitable_url(self) -> str:
        """URL мини-таблицы"""
        return self.__minitable_url

    @property
    def megatable_url(self) -> str:
        """URL мега-таблицы"""
        return self.__megatable_url

    async def load_google_sheet_to_dataframe(self, url: str) -> pd.DataFrame:
        """Загрузка google таблицы в DataFrame"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        excel_data = await response.read()
                        excel_file = BytesIO(excel_data)
                        df = pd.read_excel(excel_file, engine="openpyxl")
                        return df
                    else:
                        raise GoogleTablesException(
                            f"Ошибка при загрузке данных: статус {response.status}"
                        )
            except Exception as e:
                raise GoogleTablesException(f"Ошибка при загрузке данных из Google Sheets: {e}")

    def preprocess_megatable_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Предобработка данных мега-таблицы"""
        df.columns = df.columns.str.strip()
        for column in ["Роль", "Проект / Пространство", "Продукт / Сервис"]:
            if column in df.columns:
                df[column] = df[column].apply(lambda x: normalize_string(x))
            if column == "Роль":
                df[column] = df[column].apply(lambda x: str(x).lower())
        df["ID страницы из БЗ"] = df["ID страницы из БЗ"].apply(cast_to_int).astype(int)
        df["Проект / Пространство"] = df["Проект / Пространство"].fillna("").astype(str)
        return df
