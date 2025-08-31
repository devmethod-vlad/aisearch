from fastapi import Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.common.exceptions.exceptions import MilvusCollectionLoadTimeoutError, RequestError
from app.domain.exceptions import (
    QueueMaxSizeException,
    TimeoutException,
)


async def validation_exception_handler(
    request: Request, exception: RequestValidationError
) -> JSONResponse:
    """Обработчик ошибки валидации"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"status": "error", "message": f"Validation error, {exception.errors()}"},
    )


async def any_exception_handler(request: Request, exception: Exception) -> Response:
    """Обработчик Exception"""
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


async def milvus_collection_load_timeout_exception_handler(
    request: Request, exception: Exception
) -> Response:
    """Обработчик MilvusCollectionLoadTimeoutError"""
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


async def api_timeout_exception_handler(request: Request, exception: Exception) -> Response:
    """Обработчик ошибки таймаута на запросы"""
    return Response(status_code=status.HTTP_429_TOO_MANY_REQUESTS)


async def queue_max_size_exception_handler(request: Request, exception: Exception) -> Response:
    """Обработчик ошибки максимальной длины очереди"""
    return Response(status_code=status.HTTP_423_LOCKED)


async def request_exception_handler(request: Request, exception: RequestError) -> JSONResponse:
    """Обработчик RequestError"""
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Не удаётся совершить запрос на сторонний ресурс"},
    )


exception_config = {
    RequestValidationError: validation_exception_handler,
    Exception: any_exception_handler,
    MilvusCollectionLoadTimeoutError: milvus_collection_load_timeout_exception_handler,
    TimeoutException: api_timeout_exception_handler,
    QueueMaxSizeException: queue_max_size_exception_handler,
    RequestError: request_exception_handler,
}
