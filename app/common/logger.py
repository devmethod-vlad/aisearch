import logging
import multiprocessing
import os
from enum import Enum

from app.settings.config import settings


class LoggerType(Enum):
    """Тип логгера"""

    APP = "app"
    CELERY = "celery"
    TEST = "test"
    QUEUE = "queue"


class AISearchLogger(logging.Logger):
    """Логгер с поддержкой вывода в консоль и файл одновременно (совместим с Celery)."""

    def __init__(self, logger_type: LoggerType, name: str = __name__, level: int = logging.DEBUG):
        # важно использовать getLogger, чтобы не создать изолированный логгер
        base_logger = logging.getLogger(name)
        super().__init__(name, level)
        self.__dict__.update(base_logger.__dict__)

        self.logger_type = logger_type
        self.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )

        # --- консольный handler ---
        # добавляем только если root не имеет своих (чтобы не ломать Celery)

        is_celery_worker = multiprocessing.current_process().name.startswith("ForkPoolWorker")

        if not is_celery_worker and not any(isinstance(h, logging.StreamHandler) for h in self.handlers):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.addHandler(console_handler)

        # --- файловый handler ---
        if self.logger_type != LoggerType.TEST:
            log_dir = self._determine_logpath()
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "logs.log")

            # чтобы не было дубликатов, проверим наличие FileHandler
            if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_path)
                       for h in self.handlers):

                file_handler = logging.FileHandler(log_path, mode="a")
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                self.addHandler(file_handler)

        self.propagate = False

    # --- служебные методы ---

    def _determine_environment(self) -> str:
        """Определение окружения (docker или host)."""
        if os.path.exists("/.dockerenv") or os.getenv("DOCKER_ENVIRONMENT"):
            return "docker"
        return "host"

    def _determine_logpath(self) -> str:
        """Определение пути логирования."""
        env = self._determine_environment()

        if self.logger_type == LoggerType.APP:
            return settings.app.logs_contr_path if env == "docker" else settings.app.logs_host_path
        elif self.logger_type == LoggerType.CELERY:
            return settings.celery.logs_contr_path if env == "docker" else settings.celery.logs_host_path
        elif self.logger_type == LoggerType.QUEUE:
            return settings.celery.logs_queue_contr_path if env == "docker" else settings.celery.logs_queue_host_path


        raise TypeError(f"Неизвестный тип логгера ({self.logger_type})")
