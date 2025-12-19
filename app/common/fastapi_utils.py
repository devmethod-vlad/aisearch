import typing as tp

from fastapi import APIRouter
from fastapi.types import DecoratedCallable


class TolerantAPIRouter(APIRouter):
    """Роутер, который автоматически создает альтернативные пути со слешами,
    скрытые из документации Swagger
    """

    def add_api_route(
        self,
        path: str,
        endpoint: DecoratedCallable,
        **kwargs: tp.Any,
    ) -> None:
        super().add_api_route(path, endpoint, **kwargs)

        if path != "/" and not path.endswith("/"):
            slash_kwargs = kwargs.copy()
            slash_kwargs["include_in_schema"] = False

            if kwargs.get("include_in_schema") is True:
                kwargs["include_in_schema"] = False

            super().add_api_route(path + "/", endpoint, **slash_kwargs)
