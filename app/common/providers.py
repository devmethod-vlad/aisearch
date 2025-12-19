from dishka import Provider, Scope, provide
from fastapi import Request
from fastapi.security import HTTPAuthorizationCredentials

from app.common.auth import AccessBearer


class AuthProvider(Provider):
    """Провайдер для аутентификации."""

    @provide(scope=Scope.REQUEST)
    async def get_auth(self, request: Request) -> HTTPAuthorizationCredentials | None:
        """Получение аутентификации."""
        bearer = AccessBearer()
        return await bearer(request)
