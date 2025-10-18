from dishka import Provider, provide, Scope
from fastapi.security import HTTPAuthorizationCredentials
from fastapi import Request
from app.common.auth import AccessBearer


class AuthProvider(Provider):
    """Провайдер для аутентификации."""

    @provide(scope=Scope.REQUEST)
    async def get_auth(self, request: Request) -> HTTPAuthorizationCredentials | None:
        """Получение аутентификации."""
        bearer = AccessBearer()
        return await bearer(request)

