import asyncio
import functools
import gc
import importlib
import logging
import os
import sys
import typing as tp
from datetime import timezone

import pytz
from dotenv import load_dotenv
from pydantic import BaseModel


def cleanup_resources(
    logger: logging.Logger, *variables: tp.Any, clear_gpu: bool = True
) -> None:
    """Очистка ресурсов с возможностью удаления указанных переменных"""
    logger.info("🧹 Очистка ресурсов ...")

    for var in variables:
        if var is not None:
            del var

    importlib.invalidate_caches()
    gc.collect()

    if clear_gpu:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU кэш очищен ✅")

    logger.info("Все ресурсы очищены ✅")


def async_retry(
    max_attempts: int = 5,
    delay: float = 3.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    logger_attr_name: str = "logger",
) -> tp.Callable[
    [tp.Callable[..., tp.Awaitable[tp.Any]]], tp.Callable[..., tp.Awaitable[tp.Any]]
]:
    """Декоратор для повторных попыток асинхронных функций

    Args:
        max_attempts: максимальное количество попыток
        delay: задержка между попытками в секундах
        exceptions: типы исключений, при которых нужно повторять
        logger_attr_name: имя атрибута логгера в self
    """

    def decorator(
        func: tp.Callable[..., tp.Awaitable[tp.Any]],
    ) -> tp.Callable[..., tp.Awaitable[tp.Any]]:
        @functools.wraps(func)
        async def wrapper(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            logger: logging.Logger | None = None
            if args:
                self_obj: tp.Any = args[0]
                logger = getattr(self_obj, logger_attr_name, None)

            last_exception: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        if logger and logger.isEnabledFor(logging.WARNING):
                            logger.warning(
                                f"Повторная попытка {attempt}/{max_attempts} для {func.__name__}"
                            )
                        await asyncio.sleep(delay)

            if logger and logger.isEnabledFor(logging.ERROR):
                logger.error(
                    f"Не удалось выполнить {func.__name__} после {max_attempts} попыток: {type(last_exception)}"
                )

            if last_exception is None:
                # Теоретически не должно произойти, но для типизации
                raise RuntimeError(
                    "Произошла неизвестная ошибка при выполнении функции"
                )

            raise last_exception

        return wrapper

    return decorator


@tp.no_type_check
def get_system_root() -> str:
    """Возвращает SYSTEMROOT из переменных окружения"""
    if sys.platform == "win32":
        load_dotenv()
        return os.environ.get("SYSTEMROOT", "C:\\Windows")
    return ""


def settings_to_env_vars(settings_obj: BaseModel) -> dict[str, str]:
    """Универсальное преобразование Settings в переменные окружения.
    Автоматически использует env_prefix из model_config.
    """
    env_vars = {}

    for field_name, field_info in settings_obj.model_fields.items():
        value = getattr(settings_obj, field_name)

        if field_name.startswith("_"):
            continue

        model_config = getattr(type(settings_obj), "model_config", {})
        prefix = ""

        if isinstance(model_config, dict) or hasattr(model_config, "get"):
            prefix = model_config.get("env_prefix", "")

        if isinstance(value, BaseModel):
            nested_vars = settings_to_env_vars(value)

            for nested_key, nested_value in nested_vars.items():
                env_vars[nested_key] = nested_value

        elif value is not None:
            var_name = f"{prefix.upper()}{field_name.upper()}"

            if isinstance(value, bool):
                env_vars[var_name] = "true" if value else "false"
            elif isinstance(value, (list, tuple)):
                env_vars[var_name] = ",".join(str(item) for item in value)
            else:
                env_vars[var_name] = str(value)

    return env_vars


def exit_with_error(logger: logging.Logger, text: str, code: int = 1) -> None:
    """Завершить работу скрипта с ошибкой"""
    logger.error(text)
    logger.info(f"Завершение pre_launch ({code})")
    sys.exit(code)


def get_timezone() -> timezone:
    return pytz.timezone(os.environ.get("TZ", "Europe/Moscow"))
