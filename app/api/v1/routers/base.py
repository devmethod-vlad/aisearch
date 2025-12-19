from dishka import FromDishka
from dishka.integrations.fastapi import inject
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, RedirectResponse, Response

from app.common.fastapi_utils import TolerantAPIRouter
from app.common.version import get_app_info
from app.infrastructure.healthchecks.milvus_check import MilvusHealthCheck
from app.infrastructure.healthchecks.opensearch_check import OpenSearchHealthCheck
from app.infrastructure.healthchecks.postgres_check import PostgresHealthCheck
from app.infrastructure.healthchecks.redis_check import RedisHealthCheck

router = TolerantAPIRouter(prefix="", tags=["base"])


@router.get("/", include_in_schema=False)
async def redirect_to_docs() -> RedirectResponse:
    """Редирект на документацию"""
    return RedirectResponse(url="/docs")


@router.options("/{path:path}", include_in_schema=False)
async def preflight_handler() -> Response:
    """Проверка"""
    return Response(status_code=200)


@router.get("/healthcheck")
@inject
async def healthcheck(
    postgres_check: FromDishka[PostgresHealthCheck] = None,
    redis_check: FromDishka[RedisHealthCheck] = None,
    opensearch_check: FromDishka[OpenSearchHealthCheck] = None,
    milvus_check: FromDishka[MilvusHealthCheck] = None,
) -> JSONResponse:
    """Проверка здоровья всех сервисов"""
    health_checks = [
        postgres_check,
        redis_check,
        opensearch_check,
        milvus_check,
    ]

    results = {}
    overall_status = "ok"

    for check in health_checks:
        try:
            result = await check.check()
            results[check.name] = result

            if result.get("status") == "error":
                overall_status = "error"
        except Exception as e:
            results[check.name] = {
                "status": "error",
                "message": f"Check failed: {str(e)}",
            }
            overall_status = "error"

    health_status = {"status": overall_status, "services": results}

    status_code = 200 if overall_status == "ok" else 503

    return JSONResponse(
        content=jsonable_encoder(health_status), status_code=status_code
    )


@router.get("/version")
async def version() -> JSONResponse:
    """Версия приложения"""
    return JSONResponse(content=jsonable_encoder(get_app_info()), status_code=200)
