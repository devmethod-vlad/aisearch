import contextlib
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Iterable
from datetime import timedelta
from typing import Any, Protocol, Union


class KeyValueClientProtocol(Protocol):
    """Протокол клиента хранилища ключ-значение."""

    async def get(self, key: str) -> Any:
        """Получение по ключу"""
        ...

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        uttl: int | None = None,
        is_exists: bool = False,
        not_exist: bool = False,
        get_prev: bool = False,
    ) -> Any:
        """:param key: Ключ
        :param value: Значение
        :param ttl: Время существования данных
        :param is_exists: Устанавливает значение для ключа, если значение уже существует
        :param not_exist: Устанавливает значение для ключа, только если значения еще не существует
        :param get_prev: Устанавливает значение для ключа, возвращает предыдущее значение, или None, если предыдущего не было
        :param uttl: Время сущеcтвования данных в unix
        :return:
        """
        ...

    async def append(
        self, key: str, *values: Union[bytes, memoryview, str, int, float]
    ) -> Union[Awaitable[int], int]:
        """:param key: Ключ
        :param values: Значения
        """
        ...

    async def prepend(
        self, key: str, *values: Union[bytes, memoryview, str, int, float]
    ) -> Union[Awaitable[int], int]:
        """:param key: Ключ
        :param values: Значения
        """
        ...

    async def delete(self, *keys: Union[bytes, str, memoryview]) -> Any:
        """:param keys: Ключи"""
        ...

    async def list_set(self, key: str, index: int, value: Any) -> None:
        """:param key: Ключ
        :param index: Индекс элемента
        :param value: Значение
        """
        ...

    async def list_range(self, key: str, start: int, end: int) -> Union[Awaitable[list], list]:
        """:param key: Ключ
        :param start: Индекс начала получаемого слайса
        :param end: Индекс конца получаемого слайса
        """
        ...

    async def multiple_get(
        self,
        keys: Union[bytes, str, memoryview, Iterable[Union[bytes, str, memoryview]]],
    ) -> Union[Awaitable, Any]:
        """:param keys: Ключи"""
        ...

    async def multiple_set(self, mapping: dict) -> None:
        """:param mapping: Словарь пар {"ключ": "значение"}"""
        ...

    async def pop(self, key: str, count: int = None) -> Any:
        """:param key: Ключ
        :param count: Количество удаляемых элементов с конца (При использовании вернет список удаленных элементов в порядке их удаления)
        """
        ...

    async def left_pop(self, key: str, count: int = None) -> Any:
        """:param key: Ключ
        :param count: Количество удаляемых элементов с начала (При использовании вернет список удаленных элементов в порядке их удаления)
        """
        ...

    async def expire(
        self,
        key: str,
        ttl: int | timedelta,
        not_exist: bool = False,
        if_exist: bool = False,
        gt: bool = False,
        lt: bool = False,
    ) -> None:
        """:param key: Ключ
        :param ttl: Время жизни ключа, может быть указано как int, так и timedelta
        :param not_exist: Устанавливает время жизни ключа, только если у ключа его не было
        :param if_exist: Устанавливает время жизни ключа, только если у ключа оно было
        :param gt: Устанавливает время жизни ключа, только если у ключа оно было и оно было меньше устанавливаемого
        :param lt: Устанавливает время жизни ключа, только если у ключа оно было и оно было больше устанавливаемого
        """
        ...

    async def list_remove(self, key: str, value: str, count: int = 0) -> Any:
        """:param key: Ключ
        :param value: Значение
        :param count: Количество удаляемых элементов. Если count > 0, удаляет count значений, начиная с начала списка,
        если count < 0, то с конца, если count = 0, то удаляет все
        """

    async def scan_iter(self, match=None, count=None, _type=None, **kwargs) -> AsyncIterator[Any]:
        """:param match: Позволяет фильтровать ключи согласно паттерну
        :param count: Подсказывает Redis количество возвращаемых ключей за одну серию
        :param _type: Фильтрует значения по определенному типу данных Redis
        """
        ...


class PipelineProtocol(KeyValueClientProtocol):
    """Протокол пайплайна"""

    pass


class KeyValueStorageProtocol(KeyValueClientProtocol):
    """Протокол хранилища ключ-значения"""

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncGenerator[PipelineProtocol, None]:  # type: ignore
        """Сессия для транзакционных операций."""
        ...
