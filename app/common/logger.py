import logging
import os
from enum import Enum

from app.settings.config import settings


class LoggerType(Enum):
    """Тип логгера"""

    APP = "app"
    CELERY = "celery"
    TEST = "test"


class AISearchLogger(logging.Logger):
    """Логгер с поддержкой вывода в консоль и файл одновременно."""

    def __init__(self, logger_type: LoggerType, name: str = __name__, level: int = logging.DEBUG):
        super().__init__(name, level)

        self.logger_type = logger_type
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.addHandler(console_handler)

        if self.logger_type != LoggerType.TEST:
            file_handler = logging.FileHandler(self._determine_logpath() + "/logs.log", mode="a")
            file_handler.setFormatter(formatter)
            self.addHandler(file_handler)

    def _determine_environment(self) -> str:
        """Определение окружения (docker или host)"""
        if os.path.exists("/.dockerenv") or os.getenv("DOCKER_ENVIRONMENT"):
            return "docker"
        else:
            return "host"

    def _determine_logpath(self) -> str:
        """Определение пути логирования"""
        env = self._determine_environment()
        if self.logger_type == LoggerType.APP:
            if env == "docker":
                return settings.app.logs_contr_path
            else:
                return settings.app.logs_host_path
        elif self.logger_type == LoggerType.CELERY:
            if env == "docker":
                return settings.celery.logs_contr_path
            else:
                return settings.celery.logs_host_path
        raise TypeError(f"Неизвестный тип логгера ({self.logger_type})")
