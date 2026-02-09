import abc
import io
import typing as tp

import numpy as np
import torch


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
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra: dict[str, tp.Any] | None = None,
    ) -> str:
        """Метод запроса к LLM"""

    @abc.abstractmethod
    async def close(self) -> None:
        """Закрытие подключения"""


class IOpenSearchAdapter(abc.ABC):
    """Интерфейс для работы с OpenSearch"""

    @abc.abstractmethod
    async def index_exists(self, index_name: str | None = None) -> bool:
        """Проверяет существование индекса"""

    @abc.abstractmethod
    async def delete_index(self, index_name: str | None = None) -> bool:
        """Удаляет индекс. Возвращает True если индекс был удален"""

    @abc.abstractmethod
    async def create_index(self, index_name: str | None = None) -> bool:
        """Создает индекс с текущей схемой. Возвращает True если индекс был создан"""

    @abc.abstractmethod
    async def build_index_with_data(self, data: list[dict[str, tp.Any]]) -> None:
        """Загружает данные в индекс (предполагает, что индекс уже создан)"""

    @abc.abstractmethod
    async def search(self, body: dict, size: int) -> list[dict]:
        """Поиск в индексе"""

    @abc.abstractmethod
    async def upsert(self, data: list[dict[str, tp.Any]]) -> None:
        """Upsert документов в OpenSearch"""

    @abc.abstractmethod
    async def fetch_existing(self, size: int = 10000) -> list[dict[str, tp.Any]]:
        """Получить все документы из индекса OpenSearch"""

    @abc.abstractmethod
    async def delete(self, ext_ids: list[str]) -> None:
        """Удаляет документы из OpenSearch по полю ext_id."""

    @abc.abstractmethod
    async def count(self) -> int:
        """Возвращает количество документов в индексе OpenSearch."""

    @abc.abstractmethod
    async def ids_exist_by_source_field(
        self,
        incoming_ext_ids: tp.Iterable[tp.Any],
        source: str = None,
        field: str = "ext_id",
        batch_size: int = 2000,
        scan_page: int = 1000,
        scroll_keepalive: str = "5m",
    ) -> tuple[list[str], list[str], list[str]]:
        """Вернёт три списка (строки):
        - found_incoming: входящие ext_id, найденные в индексе по _source[field]
        - missing_incoming: входящие ext_id, которых нет в индексе
        - extra_in_store: ext_id, которые есть в индексе, но их нет во входящих
        """

    @abc.abstractmethod
    async def delete_by_ext_ids(
        self,
        ext_ids: list[str],
        field: str = "ext_id",
        batch_size: int = 2000,
        scan_page: int = 1000,
        scroll_keepalive: str = "5m",
    ) -> int:
        """Удаляет документы по строковому полю _source[field].
        Возвращает количество удалённых документов.
        """

    @abc.abstractmethod
    async def diff_modified_by_ext_ids(
        self,
        incoming_modified: dict[str, str],
        *,
        field: str = "ext_id",
        modified_field: str = "modified_at",
        batch_size: int = 2000,
        scan_page: int = 1000,
        scroll_keepalive: str = "5m",
    ) -> list[str]:
        """Вернёт список ext_id, у которых modified_at в индексе OpenSearch отличается
        от входящего значения.
        """

    @abc.abstractmethod
    async def close(self) -> None:
        """Закрытие соединения с клиентом."""


class ICrossEncoderAdapter(abc.ABC):
    """Адаптер кросс-энкодера"""

    @abc.abstractmethod
    def rank(
        self, pairs: list[tuple[str, str]]
    ) -> list[torch.Tensor] | np.ndarray | torch.Tensor:
        """Реранжирование"""

    @abc.abstractmethod
    def rank_fast(
        self,
        pairs: list[tuple[str, str]],
        device: str = "cuda",
        batch_size: int = 128,
        max_length: int = 192,
        dtype: str = "fp16",
    ) -> list[float]:
        """Быстрое ранкирование"""

    @abc.abstractmethod
    def ce_postprocess(self, logits: list[float]) -> list[float]:
        """Преобразует логиты CE в интерпретируемые очки.
        Режим управляется self.settings.ce_score_mode:
        - "sigmoid" (по умолчанию): независимая вероятность для каждого (query, doc)
        - "softmax": распределение по кандидатовому списку (с температурой)
        """


class IEduAdapter(abc.ABC):
    """Адаптер edu"""

    @abc.abstractmethod
    async def get_attachment_id_from_edu(
        self, filename: str, page_id: str | None = None
    ) -> str:
        """Получение id вложения на EDU"""

    @abc.abstractmethod
    async def download_vio_base_file(self) -> io.BytesIO:
        """Загрузить ВИО"""

    @abc.abstractmethod
    async def download_kb_base_file(self) -> io.BytesIO:
        """Загрузить КБ"""

    @abc.abstractmethod
    async def provoke_harvest_to_edu(self, harvest_type: str) -> bool:
        """Обновление файлов на edu"""
