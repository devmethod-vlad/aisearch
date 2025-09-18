from __future__ import annotations

import asyncio
import contextlib
import time
import typing as tp
import uuid
from contextlib import asynccontextmanager

from redis.asyncio import Redis
from redis.exceptions import WatchError

from app.infrastructure.adapters.interfaces import IRedisSemaphore
from app.settings.config import Settings


class RedisSemaphore(IRedisSemaphore):
    """Распределённый семафор на Redis
      - ZSET с элементами holder -> expiration_ms
      - try_acquire: очищает протухшие, пробует добавить holder в транзакции WATCH/MULTI
      - release: ZREM(holder)
      - heartbeat: ZADD XX(holder, now+ttl)
    Применяется глобально: для поисковых запросов и генерации.
    """

    def __init__(self, redis: Redis, settings: Settings):
        self.redis = redis
        self.key = settings.llm_global_sem.key
        self.limit = settings.llm_global_sem.limit
        self.ttl_ms = settings.llm_global_sem.ttl_ms
        self.wait_ms = settings.llm_global_sem.wait_timeout_ms
        self.hb_ms = settings.llm_global_sem.heartbeat_ms

    async def _now_ms(self) -> int:
        """Получение текущего времени в мс"""
        return int(time.time() * 1000)

    async def try_acquire(self, holder: str) -> bool:
        """Попытка захвата слота в семафоре"""
        now = await self._now_ms()
        # очистка протухших
        await self.redis.zremrangebyscore(self.key, "-inf", now)
        # оптимистичная блокировка
        async with self.redis.pipeline() as pipe:
            try:
                await pipe.watch(self.key)
                count = await pipe.zcard(self.key) or 0
                if count < self.limit:
                    exp = now + self.ttl_ms
                    pipe.multi()  # type: ignore
                    await pipe.zadd(self.key, {holder: exp}, nx=True)
                    await pipe.execute()
                    return True
                await pipe.unwatch()  # type: ignore
                return False
            except WatchError:
                return False

    async def heartbeat(self, holder: str) -> None:
        """ПРодление жизни держателя слота в семафоре"""
        now = await self._now_ms()
        exp = now + self.ttl_ms
        await self.redis.zadd(self.key, {holder: exp}, xx=True)

    async def release(self, holder: str) -> None:
        """Освобождение слота в семафоре"""
        await self.redis.zrem(self.key, holder)

    @asynccontextmanager
    async def acquire(
        self, *, timeout_ms: int | None = None, heartbeat: bool = True
    ) -> tp.AsyncGenerator[tp.Any, None]:
        """Захват семафора"""
        holder = str(uuid.uuid4())
        to = self.wait_ms if timeout_ms is None else timeout_ms
        start = await self._now_ms()
        while True:
            if await self.try_acquire(holder):
                break
            if to == 0:
                raise TimeoutError("global semaphore: limit reached")
            if (await self._now_ms()) - start >= to:
                raise TimeoutError("global semaphore: acquire timeout")
            await asyncio.sleep(0.2)

        hb_task = None
        if heartbeat and self.hb_ms > 0:

            async def _hb() -> None:
                try:
                    while True:
                        await asyncio.sleep(self.hb_ms / 1000)
                        await self.heartbeat(holder)
                except asyncio.CancelledError:
                    pass

            hb_task = asyncio.create_task(_hb())

        try:
            yield
        finally:
            if hb_task:
                hb_task.cancel()
                with contextlib.suppress(Exception):
                    await hb_task
            await self.release(holder)
