import contextlib
import typing as tp
from datetime import timedelta

from redis import Redis as SyncRedis
from redis.client import Pipeline as SyncPipeline


class SyncRedisClientMethods:
    """Методы клиента Redis для синхронного подключения"""

    client: SyncRedis | SyncPipeline

    def get(self, key: str) -> tp.Any:
        """Получение записи по ключу"""
        value = self.client.get(key)
        if value is not None:
            return value.decode()
        return None

    def set(
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
        return self.client.set(
            key, value, ex=ttl, exat=uttl, xx=is_exists, nx=not_exist, get=get_prev
        )

    def delete(self, *keys) -> tp.Any:
        """Удаление записей по ключам"""
        return self.client.delete(*keys)

    def append(self, key: str, *values: tp.Union[bytes, memoryview, str, int, float]) -> int:
        """Добавление записи в список в конец по ключу"""
        return self.client.rpush(key, *values)

    def prepend(self, key: str, *values: tp.Union[bytes, memoryview, str, int, float]) -> int:
        """Добавление записи в список в начало по ключу"""
        return self.client.lpush(key, *values)

    def list_set(self, key: str, index: int, value: tp.Any) -> str:
        """Устанавливает значение элемента списка по индексу"""
        return self.client.lset(name=key, index=index, value=value)

    def list_range(self, key: str, start: int, end: int) -> list[str] | None:
        """Получает диапазон элементов списка"""
        list_of_values = self.client.lrange(key, start, end)
        if list_of_values:
            return [value.decode() for value in list_of_values]
        return None

    def multiple_get(
        self,
        keys: tp.Union[bytes, str, memoryview, tp.Iterable[tp.Union[bytes, str, memoryview]]],
    ) -> list[str] | None:
        """Получает значения для нескольких ключей"""
        values = self.client.mget(*keys)
        if values:
            return [value.decode() for value in values if value is not None]
        return None

    def multiple_set(self, mapping: dict) -> None:
        """Устанавливает несколько ключей одновременно"""
        return self.client.mset(mapping)

    def pop(self, key: str, count: int = None) -> tp.Any:
        """Удаляет и возвращает элементы с конца списка"""
        result = self.client.rpop(key, count=count)
        if result:
            if isinstance(result, bytes):
                return result.decode()
            else:
                return [x.decode() for x in result]
        return None

    def left_pop(self, key: str, count: int = None) -> tp.Any:
        """Удаляет и возвращает элементы с начала списка"""
        result = self.client.lpop(key, count=count)
        if result:
            if isinstance(result, bytes):
                return result.decode()
            else:
                return [x.decode() for x in result]
        return None

    def expire(
        self,
        key: str,
        ttl: int | timedelta,
        not_exist: bool = False,
        if_exist: bool = False,
        gt: bool = False,
        lt: bool = False,
    ) -> None:
        """Устанавливает время жизни ключа"""
        return self.client.expire(name=key, time=ttl, nx=not_exist, xx=if_exist, gt=gt, lt=lt)

    def list_remove(self, key: str, value: str, count: int = 0) -> int | None:
        """Удаляет элементы из списка по значению"""
        return self.client.lrem(name=key, count=count, value=value)

    def scan_iter(
        self,
        match: str | None = None,
        count: tp.Union[bytes, str, memoryview, None] = None,
        _type: str | None = None,
        **kwargs: tp.Any
    ) -> tp.Iterator[tp.Any]:
        """Итератор по ключам с фильтрацией"""
        return self.client.scan_iter(match=match, count=count, _type=_type, **kwargs)

    def set_expire_time(self, key: str, ttl: int) -> None:
        """Устанавливает время жизни для ключа"""
        return self.client.expire(key, ttl)


class SyncRedisPipeline(SyncRedisClientMethods):
    """Пайплайн Redis"""

    def __init__(self, client: SyncPipeline):
        self.client = client


class SyncRedisStorage(SyncRedisClientMethods):
    """Хранилище Redis"""

    def __init__(self, client: SyncRedis):
        self.client = client

    @contextlib.contextmanager
    def session(self) -> tp.Generator[SyncPipeline, None, None]:
        """Транзакция в Redis"""
        pipeline = self.client.pipeline(transaction=True)
        yield SyncRedisPipeline(pipeline)
        pipeline.execute()
        pipeline.close()
