import abc
import typing as tp

class ILLMQueue(abc.ABC):
    """Интерфейс очереди"""

    @abc.abstractmethod
    async def enqueue(self, payload: dict[str, tp.Any]) -> tuple[str, int]:
        """Постановка в очередь задачи"""

    @abc.abstractmethod
    async def set_running(self, ticket_id: str, task_id: str) -> None:
        """Установка задачи в статус running"""

    @abc.abstractmethod
    async def set_done(self, ticket_id: str) -> None:
        """Установка задачи в статус done"""

    @abc.abstractmethod
    async def set_failed(self, ticket_id: str, error: str) -> None:
        """Установка задачи в failed"""

    @abc.abstractmethod
    async def dequeue(self) -> tuple[str, dict[str, tp.Any]] | None:
        """Извлечение из очереди"""

    @abc.abstractmethod
    async def status(self, ticket_id: str) -> dict[str, tp.Any]:
        """Получение статуса задачи"""

    @abc.abstractmethod
    async def dequeue_blocking(self, timeout: float = 0.2) -> tuple[str, dict[str, tp.Any]] | None:
        """Атомарно: BRPOPLPUSH main -> processing и возврат payload"""

    @abc.abstractmethod
    async def sweep_processing(self, stale_sec: int = 60) -> int:
        """Вернуть из processing устаревшие задачи; вернуть кол-во переставленных."""

    @abc.abstractmethod
    async def requeue(self, ticket_id: str, *, reason: str | None = None) -> None:
        """Переместить тикет из processing обратно в хвост основной очереди."""

    @abc.abstractmethod
    async def ack(self, ticket_id: str) -> None:
        """Удалить тикет из processing (LREM)"""