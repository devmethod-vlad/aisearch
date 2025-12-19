import logging
import os
from logging.config import dictConfig
from pathlib import Path

from app.settings.config import settings


def setup_logging() -> None:
    """Настраивает логирование через dictConfig для APP, Celery, Queue, Updater и Warnings."""
    app_log_level = settings.app.log_level
    app_logs_path = settings.app.logs_path
    app_logs_access_path = settings.app.logs_access_path

    celery_log_level = settings.celery.log_level
    celery_logs_path = settings.celery.logs_path

    queue_log_level = settings.celery.log_queue_level
    queue_logs_path = settings.celery.logs_queue_path

    updater_log_level = settings.extract_edu.log_level
    updater_logs_path = settings.extract_edu.logs_path

    handlers_config = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
    }

    app_handlers = ["console"]
    app_access_handlers = ["console"]

    if app_logs_path is not None:
        os.makedirs(os.path.dirname(app_logs_path), exist_ok=True)
        handlers_config["app_file"] = {
            "class": "logging.handlers.WatchedFileHandler",
            "formatter": "default",
            "filename": app_logs_path,
        }
        app_handlers.append("app_file")

    if app_logs_access_path is not None:
        os.makedirs(os.path.dirname(app_logs_access_path), exist_ok=True)
        handlers_config["access_file"] = {
            "class": "logging.handlers.WatchedFileHandler",
            "formatter": "access",
            "filename": app_logs_access_path,
        }
        app_access_handlers.append("access_file")

    celery_handlers = ["console"]

    if celery_logs_path is not None:
        os.makedirs(os.path.dirname(celery_logs_path), exist_ok=True)
        handlers_config["celery_file"] = {
            "class": "logging.handlers.WatchedFileHandler",
            "formatter": "celery_formatter",
            "filename": celery_logs_path,
        }
        celery_handlers.append("celery_file")

    queue_handlers = ["console"]

    if queue_logs_path is not None:
        os.makedirs(os.path.dirname(queue_logs_path), exist_ok=True)
        handlers_config["queue_file"] = {
            "class": "logging.handlers.WatchedFileHandler",
            "formatter": "celery_formatter",
            "filename": queue_logs_path,
        }
        queue_handlers.append("queue_file")

    updater_handlers = ["console"]

    if updater_logs_path is not None:
        os.makedirs(os.path.dirname(updater_logs_path), exist_ok=True)
        handlers_config["updater_file"] = {
            "class": "logging.handlers.WatchedFileHandler",
            "formatter": "default",
            "filename": updater_logs_path,
        }
        updater_handlers.append("updater_file")

    all_handlers_for_warnings = []
    handler_sources = [app_handlers, celery_handlers, queue_handlers, updater_handlers]

    for handler_list in handler_sources:
        for handler in handler_list:
            if handler not in all_handlers_for_warnings:
                all_handlers_for_warnings.append(handler)

    loggers_config = {
        "gunicorn.error": {
            "handlers": app_handlers,
            "level": app_log_level.upper(),
            "propagate": False,
        },
        "gunicorn.access": {
            "handlers": app_access_handlers,
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.error": {
            "handlers": app_handlers,
            "level": app_log_level.upper(),
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": app_access_handlers,
            "level": "INFO",
            "propagate": False,
        },
        "app": {
            "handlers": app_handlers,
            "level": app_log_level.upper(),
            "propagate": False,
        },
        "gunicorn": {
            "handlers": app_handlers,
            "level": app_log_level.upper(),
            "propagate": False,
        },
        "celery": {
            "handlers": celery_handlers,
            "level": celery_log_level.upper(),
            "propagate": False,
        },
        "celery.task": {
            "handlers": celery_handlers,
            "level": celery_log_level.upper(),
            "propagate": False,
        },
        "celery.redirected": {
            "handlers": celery_handlers,
            "level": celery_log_level.upper(),
            "propagate": False,
        },
        "queue": {
            "handlers": queue_handlers,
            "level": queue_log_level.upper(),
            "propagate": False,
        },
        "updater": {
            "handlers": updater_handlers,
            "level": updater_log_level.upper(),
            "propagate": False,
        },
        "apscheduler": {
            "handlers": updater_handlers,
            "level": updater_log_level.upper(),
            "propagate": False,
        },
        "py.warnings": {
            "handlers": all_handlers_for_warnings,
            "level": "WARNING",
            "propagate": False,
        },
    }

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s [%(process)d] [%(levelname)s]: %(message)s"
                },
                "access": {"format": "%(asctime)s [%(process)d] %(message)s"},
                "celery_formatter": {
                    "format": "%(asctime)s [%(process)d] [%(levelname)s] [%(name)s]: %(message)s"
                },
            },
            "handlers": handlers_config,
            "loggers": loggers_config,
        }
    )

    logging.captureWarnings(True)

    if app_logs_path is not None:
        app_logger = logging.getLogger("app")
        app_logger.info(
            f"Логирование APP настроено. Уровень: {app_log_level.upper()}, Файл: {Path(app_logs_path)}"
        )
    if celery_logs_path is not None:
        celery_logger = logging.getLogger("celery")
        celery_logger.info(
            f"Логирование Celery настроено. Уровень: {celery_log_level.upper()}, Путь: {Path(celery_logs_path)}"
        )
    if queue_logs_path is not None:
        queue_logger = logging.getLogger("queue")
        queue_logger.info(
            f"Логирование Queue Worker настроено. Уровень: {queue_log_level.upper()}, Путь: {Path(queue_logs_path)}"
        )
    if updater_logs_path is not None:
        updater_logger = logging.getLogger("updater")
        updater_logger.info(
            f"Логирование Updater Worker настроено. "
            f"Уровень: {updater_log_level.upper()}, Путь: {Path(updater_logs_path)}"
        )

    if all(
        path is None
        for path in [
            app_logs_path,
            app_logs_access_path,
            celery_logs_path,
            queue_logs_path,
            updater_logs_path,
        ]
    ):
        warning_logger = logging.getLogger("app")
        warning_logger.warning(
            "Ни APP, ни Celery, ни Queue, ни Updater логи не настроены на файлы. Логи будут только в консоли."
        )
    elif app_logs_access_path is None:
        info_logger = logging.getLogger("app")
        info_logger.info("Логи доступа (access) не настроены на файл.")

    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.info(
        f"Логирование Python warnings настроено на handlers: {all_handlers_for_warnings}"
    )
