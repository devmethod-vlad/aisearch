from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, RedirectResponse, Response

from app.common.version import get_app_info

router = APIRouter(prefix="", tags=["base"])


@router.get("/", include_in_schema=False)
async def redirect_to_docs() -> RedirectResponse:
    """Редирект на документацию"""
    return RedirectResponse(url="/docs")


@router.options("/{path:path}", include_in_schema=False)
async def preflight_handler() -> Response:
    """Проверка"""
    return Response(status_code=200)


@router.get("/healthcheck")
async def healthcheck() -> JSONResponse:
    """Проверка здоровья"""
    return JSONResponse(content=jsonable_encoder({"status": "ok"}), status_code=200)


@router.get("/version")
async def version() -> JSONResponse:
    """Версия приложения"""
    return JSONResponse(content=jsonable_encoder(get_app_info()), status_code=200)
