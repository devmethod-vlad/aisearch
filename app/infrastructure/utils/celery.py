from celery.result import AsyncResult


def async_result_to_dict(async_result: AsyncResult) -> dict:
    """Преобразует объект AsyncResult в словарь"""
    return {
        "id": async_result.id,
        "status": async_result.status,
        "state": async_result.state,
        "ready": async_result.ready(),
        "successful": async_result.successful(),
        "failed": async_result.failed(),
        "result": async_result.result if async_result.ready() else None,
        "traceback": async_result.traceback if async_result.failed() else None,
    }
