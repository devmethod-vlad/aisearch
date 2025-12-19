import abc
from types import TracebackType


class BaseAbstractUnitOfWork(abc.ABC):
    """Интерфейс базового UOW"""

    async def commit(self) -> None:
        """Commit"""
        await self._commit()

    @abc.abstractmethod
    async def rollback(self) -> None:
        """Rollback"""
        ...

    @abc.abstractmethod
    async def expunge_all(self) -> None:
        """Remove all objects from uow."""
        ...

    @abc.abstractmethod
    async def _commit(self) -> None:
        """Private commit."""
        ...

    async def __aenter__(self) -> None:  # noqa: B027
        """Enter context."""
        pass

    async def __aexit__(
        self,
        exc_t: type[BaseException] | None,
        exc_v: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context."""
        await self.rollback()
