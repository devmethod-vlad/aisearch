import os
from logging.config import dictConfig
from app.settings.config import settings

# --- Определяем директорию логов через Pydantic settings ---
log_dir = settings.app.logs_contr_path  # для app-логов
log_level = settings.app.log_level
os.makedirs(log_dir, exist_ok=True)

dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(asctime)s [%(process)d] [%(levelname)s] %(name)s: %(message)s"},
        "access": {"format": "%(asctime)s [%(process)d] %(message)s"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "default", "stream": "ext://sys.stdout"},
        "error_file": {
            "class": "logging.handlers.WatchedFileHandler",
            "formatter": "default",
            "filename": os.path.join(log_dir, "error.log"),
        },
        "access_file": {
            "class": "logging.handlers.WatchedFileHandler",
            "formatter": "access",
            "filename": os.path.join(log_dir, "access.log"),
        },
    },
    "loggers": {
        "gunicorn.error": {
            "handlers": ["console", "error_file"],
            "level": log_level.upper(),
            "propagate": False,
        },
        "gunicorn.access": {
            "handlers": ["console", "access_file"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.error": {"handlers": ["console", "error_file"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["console", "access_file"], "level": "INFO", "propagate": False},
    },
})

# Пути для Gunicorn остаются None, dictConfig уже настроен
accesslog = None
errorlog = None