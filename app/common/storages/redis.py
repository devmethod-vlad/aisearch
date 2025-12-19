import contextlib
import typing as tp
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Iterable
from datetime import timedelta

import redis
from redis.asyncio import Redis
from redis.asyncio.client import Pipeline

from app.common.storages.interfaces import KeyValueClientProtocol


class RedisClientMethods(KeyValueClientProtocol):
    """Методы клиента Redis"""

    client: redis.Redis | Pipeline

    async def get(self, key: str) -> tp.Any:
        """Получение записи по ключу"""
        value = await self.client.get(key)
        if value is not None:
            return value.decode()
        return None

    async def set(
        self,
        key: str,
        value: tp.Any,
        ttl: int | None = None,
        uttl: int | None = None,
        is_exists: bool = False,
        not_exist: bool = False,
        get_prev: bool = False,
    ) -> tp.Any:
        """Запись по ключу в Redis"""
        return await self.client.set(
            key, value, ex=ttl, exat=uttl, xx=is_exists, nx=not_exist, get=get_prev
        )

    async def delete(self, *keys: bytes | str | memoryview) -> tp.Any:
        """Удаление записей по ключам"""
        return await self.client.delete(*keys)

    async def append(
        self, key: str, *values: bytes | memoryview | str | int | float
    ) -> Awaitable[int] | int:
        """Добавление записи в список в конец по ключу"""
        return await self.client.rpush(key, *values)

    async def prepend(
        self, key: str, *values: bytes | memoryview | str | int | float
    ) -> Awaitable[int] | int:
        """Добавление записи в список в начало по ключу"""
        return await self.client.lpush(key, *values)

    async def list_set(
        self, key: str, index: int, value: tp.Any
    ) -> Awaitable[str] | str:
        """Устанавливает значение элемента списка по индексу"""
        return await self.client.lset(name=key, index=index, value=value)

    async def list_range(
        self, key: str, start: int, end: int
    ) -> Awaitable[list] | list | None:
        """Получает диапазон элементов списка"""
        list_of_values = await self.client.lrange(key, start, end)
        if list_of_values:
            return [value.decode() for value in list_of_values]
        return None

    async def multiple_get(
        self,
        keys: bytes | str | memoryview | Iterable[bytes | str | memoryview],
    ) -> Awaitable | tp.Any | None:
        """Получает значения для нескольких ключей"""
        values = await self.client.mget(*keys)
        if values:
            return [value.decode() for value in values]
        return None

    async def multiple_set(self, mapping: dict) -> None:
        """Устанавливает несколько ключей одновременно"""
        return await self.client.mset(mapping)

    async def pop(self, key: str, count: int = None) -> tp.Any:
        """Удаляет и возвращает элементы с конца списка"""
        result = await self.client.rpop(key, count=count)
        if result:
            if isinstance(result, bytes):
                return result.decode()
            else:
                return [x.decode() for x in result]
        return None

    async def left_pop(self, key: str, count: int = None) -> tp.Any:
        """Удаляет и возвращает элементы с начала списка"""
        result = await self.client.lpop(key, count=count)
        if result:
            if isinstance(result, bytes):
                return result.decode()
            else:
                return [x.decode() for x in result]
        return None

    async def expire(
        self,
        key: str,
        ttl: int | timedelta,
        not_exist: bool = False,
        if_exist: bool = False,
        gt: bool = False,
        lt: bool = False,
    ) -> None:
        """Устанавливает время жизни ключа"""
        return await self.client.expire(
            name=key, time=ttl, nx=not_exist, xx=if_exist, gt=gt, lt=lt
        )

    async def list_remove(self, key: str, value: str, count: int = 0) -> int | None:
        """Удаляет элементы из списка по значению"""
        return await self.client.lrem(name=key, count=count, value=value)

    async def scan_iter(
        self,
        match: str | None = None,
        count: bytes | str | memoryview | None = None,
        _type: str | None = None,
        **kwargs: tp.Any,
    ) -> AsyncIterator[tp.Any]:
        """Итератор по ключам с фильтрацией"""
        return self.client.scan_iter(match=match, count=count, _type=_type, **kwargs)

    async def set_expire_time(self, key: str, ttl: int) -> None:
        """Устанавливает время жизни для ключа"""
        return self.client.expire(key, ttl)

    async def hash_get(self, key: str, field: str) -> str | None:
        """Получает значение поля из хэша"""
        value = await self.client.hget(key, field)
        if value is not None:
            return value.decode()
        return None

    async def hash_set(
        self, key: str, mapping: dict[str, tp.Any] | None = None, **fields: tp.Any
    ) -> int:
        """Устанавливает поля в хэш"""
        return await self.client.hset(key, mapping=mapping or {}, **fields)

    async def hash_del(self, key: str, *fields: str) -> int:
        """Удаляет поля из хэша"""
        return await self.client.hdel(key, *fields)

    async def hash_getall(self, key: str) -> dict[str, str]:
        """Получает все поля и значения из хэша"""
        result = await self.client.hgetall(key)
        if result:
            return {k.decode(): v.decode() for k, v in result.items()}
        return {}

    async def hash_exists(self, key: str, field: str) -> bool:
        """Проверяет, существует ли поле в хэше"""
        return await self.client.hexists(key, field)

    async def hash_keys(self, key: str) -> list[str]:
        """Получает все ключи (поля) из хэша"""
        keys = await self.client.hkeys(key)
        if keys:
            return [k.decode() for k in keys]
        return []

    async def hash_vals(self, key: str) -> list[str]:
        """Получает все значения из хэша"""
        values = await self.client.hvals(key)
        if values:
            return [v.decode() for v in values]
        return []

    async def hash_len(self, key: str) -> int:
        """Получает количество полей в хэше"""
        return await self.client.hlen(key)


class RedisPipeline(RedisClientMethods):
    """Пайплайн Redis"""

    def __init__(self, client: Pipeline):
        self.client = client


class RedisStorage(RedisClientMethods):
    """Хранилище Redis"""

    def __init__(self, client: Redis):
        self.client = client

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncGenerator[RedisPipeline, None]:
        """Транзакция в Redis"""
        pipeline = await self.client.pipeline(transaction=True)
        yield RedisPipeline(pipeline)
        await pipeline.execute()
        await pipeline.aclose()
