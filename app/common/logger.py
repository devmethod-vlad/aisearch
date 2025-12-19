import logging
from enum import Enum


class LoggerType(Enum):
    """Тип логгера"""

    APP = "app"
    CELERY = "celery"
    TEST = "test"
    QUEUE = "queue"
    UPDATER = "updater"

    @property
    def logger_name(self) -> str:
        """Возвращает имя логгера, соответствующее типу."""
        return self.value


class AISearchLogger(logging.Logger):
    """Логгер, который использует стандартную иерархию Python logging,
    но инициализируется с именем, соответствующим типу (app, celery).
    """

    def __new__(
        cls, logger_type: LoggerType, name: str = __name__, level: int = logging.DEBUG
    ):
        """Создает и возвращает стандартный logging.Logger, имя которого
        определяется по logger_type. Этот метод фактически выбирает
        уже существующий логгер из иерархии Python logging.
        """
        logger_name = logger_type.logger_name
        standard_logger = logging.getLogger(logger_name)
        return standard_logger

    def __init__(
        self, logger_type: LoggerType, name: str = __name__, level: int = logging.DEBUG
    ):
        """Инициализирует атрибуты возвращённого __new__ объекта logging.Logger.
        Этот метод вызывает logging.Logger.__init__ с переданными параметрами,
        чтобы удовлетворить требования dishka к аннотациям типов.
        Фактическое поведение логгера (обработчики, уровень) определяется
        через logging.config.dictConfig и иерархию логгеров Python.
        """
        super().__init__(name=name, level=level)
