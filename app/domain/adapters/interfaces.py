import abc
from io import BytesIO

import pandas as pd

from app.domain.schemas.knowledge_base import KnowledgeError


class IConfluenceAdapter(abc.ABC):
    """Интерфейс адаптера Confluence"""

    @property
    @abc.abstractmethod
    def confluence_url(self) -> str:
        """Confluence URL"""

    @abc.abstractmethod
    async def get_html_content_from_page(self, page_id: str) -> str:
        """Получение HTML из страницы Confluence"""

    @abc.abstractmethod
    def get_knowledge_base_from_html(
        self, html_content: str, page_id: str, filter_actual: bool, filter_for_user: bool
    ) -> tuple[pd.DataFrame, list[KnowledgeError]]:
        """Получение контента БЗ из HTML"""

    @abc.abstractmethod
    def preprocess_knowledge_base_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Предобработка данных БЗ"""


class IGoogleTablesAdapter(abc.ABC):
    """Интерфейс адаптера Google-таблиц"""

    @property
    @abc.abstractmethod
    def minitable_url(self) -> str:
        """URL мини-таблицы"""

    @property
    @abc.abstractmethod
    def megatable_url(self) -> str:
        """URL мега-таблицы"""

    @abc.abstractmethod
    async def load_google_sheet_to_dataframe(self, url: str) -> pd.DataFrame:
        """Загрузка google таблицы в DataFrame"""

    @abc.abstractmethod
    def preprocess_megatable_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Предобработка данных мега-таблицы"""


class IEduAdapter(abc.ABC):
    """Интерфейс адаптера EDU"""

    @abc.abstractmethod
    async def create_or_update_file_on_edu(self, filename: str, file_data: BytesIO) -> None:
        """Создание/обновление файла на EDU"""

    @abc.abstractmethod
    async def get_attachment_id_from_edu(self, filename: str) -> str:
        """Получение id вложения на EDU"""
