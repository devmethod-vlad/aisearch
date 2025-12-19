import datetime
import logging
import os
import sys
import time
import traceback

import redis

from app.common.logger import AISearchLogger, LoggerType
from app.common.storages.sync_redis import SyncRedisStorage
from app.infrastructure.utils.process import (
    get_current_btime,
    get_process_absolute_starttime,
    get_worker_process_keys,
)
from app.settings.config import settings


def get_process_grace_period() -> int:
    """Получает значение grace period для процессов из переменных окружения."""
    grace_period_str = os.getenv("CELERY_GRACE_PERIOD_SECONDS")
    if not grace_period_str:
        raise ValueError("CELERY_GRACE_PERIOD_SECONDS environment variable is not set")

    return int(grace_period_str)


def setup_logger() -> logging.Logger:
    """Настраивает и возвращает логгер."""
    prefix = "[search_worker/healthcheck.py]"

    logger = AISearchLogger(logger_type=LoggerType.CELERY)

    original_handlers = logger.handlers.copy()
    for handler in original_handlers:
        logger.removeHandler(handler)

    celery_logs_path = os.getenv("CELERY_LOGS_PATH")
    if not celery_logs_path:
        raise ValueError

    os.makedirs(os.path.dirname(celery_logs_path), exist_ok=True)

    class CustomFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            timestamp = datetime.datetime.now(tz=datetime.UTC).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            level = record.levelname
            return f"{timestamp} {prefix} [{level}] {record.getMessage()}"

    file_handler = logging.FileHandler(celery_logs_path)
    file_handler.setFormatter(CustomFormatter())
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(CustomFormatter())
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def healthcheck() -> None:
    """Основная функция healthcheck для Search Worker."""
    logger = setup_logger()

    # 1. Получаем worker_id
    worker_id = os.getenv("WORKER_ID")
    if not worker_id:
        logger.error("💥 ERROR: WORKER_ID environment variable is not set")
        sys.exit(1)

    # 2. Подключаемся к Redis
    redis_client = SyncRedisStorage(
        client=redis.from_url(settings.redis.dsn, decode_responses=True)
    )

    # 3. Получаем ключи процессов из Redis
    process_keys = get_worker_process_keys(redis_client, worker_id)

    # 4. Если ключей нет - Celery еще не запустился
    if not process_keys:
        logger.info("⏳ Waiting for Celery startup...")
        sys.exit(1)

    # 5. Получаем grace period для процессов
    process_grace_period_seconds = get_process_grace_period()

    # 6. Получаем текущее время системы
    current_system_time = time.time()

    # 7. Переменные для сбора информации
    any_process_old = False
    all_processes_healthy = True
    errors_detected = False
    unhealthy_old_processes = []

    # 8. Собираем информацию о всех процессах
    for process_key in process_keys:
        try:
            pid_str = redis_client.client.hget(process_key, "pid")
            all_healthy = redis_client.client.hget(process_key, "all_healthy")
            proc_created_at = redis_client.client.hget(process_key, "proc_created_at")

            if not pid_str or not proc_created_at:
                logger.error(f"❌ Missing process data in Redis key: {process_key}")
                errors_detected = True
                all_processes_healthy = False
                continue

            pid = int(pid_str)
            is_healthy = bool(int(all_healthy)) if all_healthy else False

            if not is_healthy:
                all_processes_healthy = False

            try:
                process_absolute_start = get_process_absolute_starttime(pid)
                process_age = current_system_time - process_absolute_start

                if process_age >= process_grace_period_seconds:
                    any_process_old = True

                    if not is_healthy:
                        process_name = (
                            process_key.split(":")[-1]
                            if ":" in process_key
                            else process_key
                        )
                        unhealthy_old_processes.append(
                            {
                                "pid": pid,
                                "process_name": process_name,
                                "age": process_age,
                                "key": process_key,
                            }
                        )
                else:
                    remaining = process_grace_period_seconds - process_age
                    status = "healthy" if is_healthy else "warming up"
                    logger.info(
                        f"⏳ PID {pid}: {process_age:.1f}s / {process_grace_period_seconds}s "
                        f"({status}, {remaining:.1f}s remaining)"
                    )

            except FileNotFoundError:
                try:
                    proc_created_at_float = float(proc_created_at)
                    btime = get_current_btime()
                    process_absolute_creation = btime + proc_created_at_float
                    process_age = current_system_time - process_absolute_creation

                    logger.error(
                        f"❌ PID {pid}: process not found (age: {process_age:.1f}s), deleting Redis key: {process_key}"
                    )
                    redis_client.client.delete(process_key)
                    errors_detected = True
                    all_processes_healthy = False

                except Exception as e:
                    logger.error(
                        f"❌ PID {pid}: error calculating process age ({type(e)}): {traceback.format_exc()}"
                    )
                    errors_detected = True
                    all_processes_healthy = False

            except PermissionError:
                logger.error(
                    f"❌ PID {pid}: permission denied to access process, check permissions"
                )
                errors_detected = True
                all_processes_healthy = False

            except Exception as e:
                logger.error(
                    f"❌ PID {pid}: error getting process info ({type(e)}): {traceback.format_exc()}"
                )
                errors_detected = True
                all_processes_healthy = False

        except Exception as e:
            logger.error(
                f"💥 Error checking process {process_key} ({type(e)}): {traceback.format_exc()}"
            )
            errors_detected = True
            all_processes_healthy = False

    # 9. Теперь обрабатываем собранную информацию
    if errors_detected:
        # Были критические ошибки
        sys.exit(1)

    # Используем elif для более читаемой структуры
    if not any_process_old:
        if all_processes_healthy:
            logger.info("✅ All processes healthy and within grace period")
        else:
            logger.info("⚠️ Some processes warming up, all within grace period")
            sys.exit(1)
    elif all_processes_healthy:
        # Все старые процессы здоровы - тихий успех
        pass
    elif unhealthy_old_processes:
        # Есть старые нездоровые процессы с информацией
        logger.error(
            "💥 ERROR: The following processes are unhealthy after grace period:"
        )
        for proc in unhealthy_old_processes:
            logger.error(
                f"   ❌ PID {proc['pid']} ({proc['process_name']}): "
                f"{proc['age']:.1f}s old (grace: {process_grace_period_seconds}s)"
            )
        sys.exit(1)
    else:
        # Есть старые нездоровые процессы без детальной информации
        logger.error("💥 ERROR: Some processes unhealthy after grace period")
        sys.exit(1)

    # 10. Успешное завершение
    sys.exit(0)


if __name__ == "__main__":
    healthcheck()
