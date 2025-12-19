import logging
import time
import typing as tp


def init_logger(
    name: str | None = "METRICS",
    level: int | str = logging.DEBUG,
    fmt: str = "%(asctime)s - %(levelname)s - %(message)s",
    datefmt: str | None = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """Возвращает настроенный logger для метрик.
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


def metrics_print(
    label: str, start_time: float, precision: int = 4, metrics_enabled: bool = True
) -> float:
    elapsed = time.perf_counter() - start_time
    elapsed_rounded = round(elapsed, precision)
    logger = init_logger()
    if metrics_enabled:
        # Сообщение без собственного timestamp/level — их добавит верхний форматтер Celery
        logger.info(f"{label}: {elapsed_rounded:.{precision}f} сек")
    return elapsed_rounded


def _now_ms() -> int:
    """Получение текущего времени в мс"""
    return int(time.time() * 1000)


def _convert_to_ms_or_return_0(value: tp.Any) -> int:
    """Преобразование значения в миллисекунды или возврат 0, если value не число"""
    try:
        num_value = float(value)
        return int(num_value * 1000)
    except (TypeError, ValueError, AttributeError, OverflowError):
        return 0
