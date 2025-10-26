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
    def build_index(self, data: list[dict[str, tp.Any]]) -> None:
        """Построение индекса"""

    @abc.abstractmethod
    def search(self, body: dict, size: int) -> list[dict]:
        """Поиск opensearch"""


class IBM25Adapter(abc.ABC):
    """Адаптер кросс-энкодера"""

    @staticmethod
    @abc.abstractmethod
    def build_index(
        data: list[dict[str, tp.Any]], index_path: str, texts: list[str], logger: AISearchLogger
    ) -> None:
        """Построение индекса"""

    @abc.abstractmethod
    def ensure_index(self) -> None:
        """Подгрузка индекса"""

    @abc.abstractmethod
    def search(self, query: str, top_k: int = 50) -> list[dict[str, tp.Any]]:
        """Возвращает список кандидатов"""


class ICrossEncoderAdapter(abc.ABC):
    """Адаптер кросс-энкодера"""

    @abc.abstractmethod
    def rank(self, pairs: list[tuple[str, str]]) -> list[torch.Tensor] | np.ndarray | torch.Tensor:
        """Реранжирование"""

    @abc.abstractmethod
    def rank_fast(self, pairs: list[tuple[str, str]], device: str = "cuda", batch_size: int = 128,  max_length: int = 192, dtype: str = "fp16") -> list[float]:
        """Быстрое ранкирование"""

    @abc.abstractmethod
    def ce_postprocess(self, logits: list[float]) -> list[float]:
        """
        Преобразует логиты CE в интерпретируемые очки.
        Режим управляется self.settings.ce_score_mode:
        - "sigmoid" (по умолчанию): независимая вероятность для каждого (query, doc)
        - "softmax": распределение по кандидатовому списку (с температурой)
        """
