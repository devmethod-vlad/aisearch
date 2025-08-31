from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import pandas as pd


class KnowledgeErrorType(Enum):
    """Тип ошибки при парсинге БЗ"""

    NESTED_TABLE = "Вложенная таблица"
    COLUMN_NAMES = "Некорректные названия столбцов"
    NUMBER_OF_COLUMNS = "Отличается количество столбцов"
    UNKNOWN_ATTACHMENT = "Неизвестное вложение"
    KNOWLEDGE_DUPLICATE = "Дубль знания"
    KNOWLEDGE_NUMBER_NOT_SPECIFIED = "Не указан Номер знания"
    KNOWLEDGE_NUMBER_INCORRECT_FORMAT = "Некорректный формат номера знания"
    DOWNLOAD_ATTACHMENT_FAILED = "Не удалось скачать недоступное вложение"
    MAIN_HEADER_MISSING = "Заголовок не найден"
    PAGE_NOT_FOUND = "Страница не найдена"
    MERGED_COLUMNS = "Объединенные ячейки в столбцах"
    MERGED_ROWS = "Объединенные ячейки в строках"
    SEVERAL_TABLES = "На странице несколько таблиц БЗ"
    HEADER_ROW_MISSING = "Отсутствует строка заголовков"
    DATA_MISSING = "Нет таблицы БЗ или таблица пуста"

    def __str__(self) -> str:
        """Строковое представление"""
        return self.value


@dataclass
class KnowledgeError:
    """Ошибка парсинга БЗ"""

    url: str
    error_type: str  # KnowledgeErrorType
    knowledge_number: str = ""
    column_name: str = ""
    content_from_error_desc: str = ""

    __COLUMN_NAMES: ClassVar[dict[str, str]] = {
        "url": "URL на wiki-страницу",
        "knowledge_number": "Номер знания",
        "error_type": "Тип ошибки",
        "column_name": "Название столбца с ошибкой",
        "content_from_error_desc": 'Содержимое из "Описание ошибки"',
    }

    @classmethod
    def to_dataframe(cls, errors: list["KnowledgeError"]) -> pd.DataFrame:
        """Преобразование списка ошибок в DataFrame с русскими названиями колонок"""
        df = pd.DataFrame([error.__dict__ for error in errors])
        return df.rename(columns=cls.__COLUMN_NAMES)
