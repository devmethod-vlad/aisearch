class DomainException(Exception):
    """Базовая ошибка предметной области"""


class TimeoutException(DomainException):
    """Ошибка частых запросов"""


class QueueMaxSizeException(DomainException):
    """Ошибка очереди"""


class ConfluenceException(DomainException):
    """Ошибка адаптера Confluence"""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code

    def __str__(self) -> str:
        """Строковое представление"""
        if self.status_code is not None:
            return f"[{self.status_code}] {super().__str__()}"
        return super().__str__()


class GoogleTablesException(DomainException):
    """Ошибка адаптера GoogleTables"""
