from fastapi import Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.common.exceptions.exceptions import LoginError
from app.settings.config import settings


class AccessBearer(HTTPBearer):
    """Класс для авторизации"""

    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)

    async def __call__(self, request: Request):
        """Авторизация пользователя"""
        credentials: HTTPAuthorizationCredentials | None = await super().__call__(request)
        if credentials:
            if not credentials.scheme == "Bearer":
                raise LoginError("Invalid authentication scheme")
            if not self.verify_key(credentials.credentials):
                raise LoginError("Invalid access key")
            return credentials.credentials
        else:
            raise LoginError("No credentials")

    def verify_key(self, key: str) -> bool:
        """Верификация ключа"""
        return key == settings.app.access_key
