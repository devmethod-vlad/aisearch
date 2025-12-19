from fastapi import Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic_core import PydanticCustomError

from app.common.exceptions.exceptions import (
    MilvusCollectionLoadTimeoutError,
    NotFoundError,
    RequestException,
)
from app.domain.exceptions import (
    FeedbackException,
    QueueMaxSizeException,
    TimeoutException,
)


async def validation_exception_handler(
    request: Request, exception: PydanticCustomError
) -> JSONResponse:
    """Обработчик ошибки валидации"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": jsonable_encoder(exception.errors()),
        },
    )


async def no_found_exception_handler(
    request: Request, exception: NotFoundError
) -> Response:
    """Обработчик ошибки отсуствия объекта"""
    return Response(status_code=status.HTTP_404_NOT_FOUND)


async def any_exception_handler(request: Request, exception: Exception) -> Response:
    """Обработчик Exception"""
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


async def milvus_collection_load_timeout_exception_handler(
    request: Request, exception: Exception
) -> Response:
    """Обработчик MilvusCollectionLoadTimeoutError"""
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


async def api_timeout_exception_handler(
    request: Request, exception: TimeoutException
) -> Response:
    """Обработчик ошибки таймаута на запросы"""
    return Response(status_code=status.HTTP_429_TOO_MANY_REQUESTS)


async def queue_max_size_exception_handler(
    request: Request, exception: QueueMaxSizeException
) -> Response:
    """Обработчик ошибки максимальной длины очереди"""
    return Response(status_code=status.HTTP_423_LOCKED)


async def request_exception_handler(
    request: Request, exception: RequestException
) -> JSONResponse:
    """Обработчик RequestException"""
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Не удаётся совершить запрос на сторонний ресурс"},
    )


async def feedback_exception_handler(
    request: Request, exception: FeedbackException
) -> JSONResponse:
    """Обработчик FeedbackException"""
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={"detail": "Оценка уже существует / отсутствует id записи статистики"},
    )


exception_config = {
    PydanticCustomError: validation_exception_handler,
    Exception: any_exception_handler,
    MilvusCollectionLoadTimeoutError: milvus_collection_load_timeout_exception_handler,
    TimeoutException: api_timeout_exception_handler,
    QueueMaxSizeException: queue_max_size_exception_handler,
    RequestException: request_exception_handler,
    FeedbackException: feedback_exception_handler,
    NotFoundError: no_found_exception_handler,
}
