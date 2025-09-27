import abc
import typing as tp

import numpy as np
import pandas as pd
import torch

from app.common.logger import AISearchLogger


class IRedisSemaphore(abc.ABC):
    """Глобальный семафор на редисе"""

    @abc.abstractmethod
    async def try_acquire(self, holder: str) -> bool:
        """Попытка захвата семафора"""

    @abc.abstractmethod
    async def heartbeat(self, holder: str) -> None:
        """Обновление времени держателя семафора"""

    @abc.abstractmethod
    async def release(self, holder: str) -> None:
        """Освобождение места семафора"""

    @abc.abstractmethod
    async def acquire(
        self, *, timeout_ms: int | None = None, heartbeat: bool = True
    ) -> tp.AsyncGenerator[tp.Any, None]:
        """Асинхронный контекстный менеджер для захвата семафора"""


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
    async def dequeue_blocking(self, timeout: int = 0) -> tuple[str, dict[str, tp.Any]] | None:
        """Атомарно: BRPOPLPUSH main -> processing и возврат payload"""

    @abc.abstractmethod
    async def ack(self, ticket_id: str) -> None:
        """Удалить тикет из processing (LREM)"""


class IVLLMAdapter(abc.ABC):
    """Адаптер для LLM"""

    @abc.abstractmethod
    async def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: tp.Optional[int] = None,
        temperature: tp.Optional[float] = None,
        top_p: tp.Optional[float] = None,
        extra: tp.Optional[dict[str, tp.Any]] = None,
    ) -> str:
        """Метод запроса к LLM"""

    @abc.abstractmethod
    async def close(self) -> None:
        """Закрытие подключения"""


class IOpenSearchAdapter(abc.ABC):
    """Адаптер OpenSearch"""

    @abc.abstractmethod
    def build_index(self, data: pd.DataFrame) -> None:
        """Построение индекса"""

    @abc.abstractmethod
    def search(self, body: dict, size: int) -> list[dict]:
        """Поиск opensearch"""


class IBM25Adapter(abc.ABC):
    """Адаптер кросс-энкодера"""

    @staticmethod
    @abc.abstractmethod
    def build_index(
        data: pd.DataFrame, index_path: str, texts: list[str], logger: AISearchLogger
    ) -> None:
        """Построение индекса"""

    @abc.abstractmethod
    def ensure_index(self) -> None:
        """Подгрузка индекса"""

    @abc.abstractmethod
    def search(self, query: str, top_k: int = 50) -> list[dict[str, tp.Any]]:
        """Возвращает список кандидатов:
        [{"ext_id","question","analysis","answer","score_bm25","source":"bm25"}, ...]
        """


class ICrossEncoderAdapter(abc.ABC):
    """Адаптер кросс-энкодера"""

    @abc.abstractmethod
    def rank(self, pairs: list[tuple[str, str]]) -> list[torch.Tensor] | np.ndarray | torch.Tensor:
        """Реранжирование"""
