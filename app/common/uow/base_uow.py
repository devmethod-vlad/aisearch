from types import TracebackType

from app.common.database import Database
from app.common.uow.interfaces import BaseAbstractUnitOfWork


class BaseUnitOfWork(BaseAbstractUnitOfWork):
    """Базовая единица работы"""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def __aenter__(self) -> None:
        """Вход в контекст UOW"""
        await super().__aenter__()
        self.session = self.db.session_factory()

    async def __aexit__(
        self,
        exc_t: type[BaseException] | None,
        exc_v: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Выход из контекста UOW"""
        self.session.expunge_all()
        await super().__aexit__(exc_t, exc_v, exc_tb)
        await self.session.close()

    async def _commit(self) -> None:
        """Метод коммита"""
        await self.session.commit()

    async def rollback(self) -> None:
        """Метод Отката"""
        await self.session.rollback()

    async def expunge_all(self) -> None:
        """Метод очищения объектов из сессии"""
        self.session.expunge_all()
