import time
import logging
from typing import Optional

def init_logger(
    name: Optional[str] = "METRICS",
    level: int | str = logging.DEBUG,
    fmt: str = "%(asctime)s - %(levelname)s - %(message)s",
    datefmt: Optional[str] = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """
    Возвращает настроенный logger для метрик.
    В окружении Celery не навешиваем свой StreamHandler, чтобы избежать дублей.
    """
    logger = logging.getLogger(name if name is not None else __name__)
    logger.setLevel(level)

    # Если корневой логгер не настроен (скрипт вне Celery) — добавляем консольный handler.
    # В Celery root уже имеет handlers => ниже блок не выполнится, дублей не будет.
    if not logging.getLogger().handlers and not logger.handlers:
        h = logging.StreamHandler()
        h.setLevel(level)
        h.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(h)

    logger.propagate = True  # отдаём вверх на единый формат Celery
    return logger

def metrics_print(label: str, start_time: float, precision: int = 4, metrics_enabled: bool = True) -> float:
    elapsed = time.perf_counter() - start_time
    elapsed_rounded = round(elapsed, precision)
    logger = init_logger()
    if metrics_enabled:
        # Сообщение без собственного timestamp/level — их добавит верхний форматтер Celery
        logger.info(f"{label}: {elapsed_rounded:.{precision}f} сек")
    return elapsed_rounded
