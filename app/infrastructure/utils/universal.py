import asyncio
import functools
import gc
import importlib
import logging
import typing as tp


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
