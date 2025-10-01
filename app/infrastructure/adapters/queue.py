from __future__ import annotations

import json
import time
import typing as tp

from redis import WatchError
from redis.asyncio import Redis

from app.infrastructure.adapters.interfaces import ILLMQueue
from app.settings.config import Settings


class LLMQueue(ILLMQueue):
    """Очередь на Redis:
    - LIST: очередь ticket_id (FIFO)
    - HASH: мета по тикету (state, payload, task_id, error ...)
    """

    def __init__(self, redis: Redis, settings: Settings):
        self.redis = redis
        self.qkey = settings.llm_queue.queue_list_key
        self.tprefix = settings.llm_queue.ticket_hash_prefix
        self.max_size = settings.llm_queue.max_size
        self.ticket_ttl = settings.llm_queue.ticket_ttl
        self.pkey = settings.llm_queue.processing_list_key or f"{self.qkey}:processing"

    async def requeue(self, ticket_id: str, *, reason: str | None = None) -> None:
        """Снять тикет с processing и поставить его в конец основной очереди."""
        now = int(time.time())
        hkey = f"{self.tprefix}{ticket_id}"
        async with self.redis.pipeline() as pipe:
            # Обновляем статус и метаданные
            await pipe.hset(hkey, mapping={
                "state": "queued",
                "task_id": "",
                "error": reason or "",
                "updated_at": now,
            })
            await pipe.lrem(self.pkey, 1, ticket_id)
            await pipe.rpush(self.qkey, ticket_id)
            await pipe.execute()

    async def enqueue(self, payload: dict[str, tp.Any]) -> tuple[str, int]:
        """Постановка задачи в очередь с учётом позиции и защиты от переполнения."""
        ticket_id = payload["ticket_id"]
        now = int(time.time())
        hkey = f"{self.tprefix}{ticket_id}"
        data = {
            "state": "queued",
            "created_at": now,
            "updated_at": now,
            "payload": json.dumps(payload, ensure_ascii=False),
            "task_id": "",
            "error": "",
        }

        while True:
            async with self.redis.pipeline() as pipe:
                try:
                    # Следим сразу за основной и processing-очередью
                    await pipe.watch(self.qkey, self.pkey)

                    # Текущее количество элементов (ожидающих + в обработке)
                    queued_len = await pipe.llen(self.qkey) or 0
                    processing_len = await pipe.llen(self.pkey) or 0
                    total_len = queued_len + processing_len

                    if total_len >= self.max_size:
                        await pipe.unwatch()  # type: ignore
                        raise OverflowError(f"LLM queue overflow: {total_len}/{self.max_size}")

                    # Начинаем транзакцию
                    pipe.multi()  # type: ignore
                    # Записываем мета по тикету
                    await pipe.hset(hkey, mapping=data)
                    await pipe.expire(hkey, self.ticket_ttl)
                    # Ставим тикет в очередь
                    await pipe.rpush(self.qkey, ticket_id)
                    # Получаем новую длину qkey (позиция тикета)
                    await pipe.llen(self.qkey)

                    res = await pipe.execute()
                    # Последний результат — длина очереди после вставки
                    pos = int(res[-1]) + processing_len
                    return ticket_id, pos

                except WatchError:
                    # Если очередь изменилась между WATCH и EXEC → пробуем заново
                    continue

    async def set_running(self, ticket_id: str, task_id: str) -> None:
        """Установка задачи в статус running"""
        await self.redis.hset(
            f"{self.tprefix}{ticket_id}",
            mapping={"state": "running", "task_id": task_id, "updated_at": int(time.time())},
        )

    async def set_done(self, ticket_id: str) -> None:
        """Установка задачи в статус done"""
        await self.redis.hset(
            f"{self.tprefix}{ticket_id}", mapping={"state": "done", "updated_at": int(time.time())}
        )

    async def set_failed(self, ticket_id: str, error: str) -> None:
        """Установка задачи в статус failed"""
        await self.redis.hset(
            f"{self.tprefix}{ticket_id}",
            mapping={"state": "failed", "error": error, "updated_at": int(time.time())},
        )

    async def dequeue(self) -> tuple[str, dict[str, tp.Any]] | None:
        """Извлечение задачи из начала очереди"""
        ticket_id = await self.redis.lpop(self.qkey)

        if not ticket_id:
            return None
        ticket_id = ticket_id.decode()
        hkey = f"{self.tprefix}{ticket_id}"
        # data = await self.redis.hgetall(hkey)
        raw = await self.redis.hget(hkey, "payload")

        if raw is None:
            return ticket_id, {}

        payload = (
            json.loads(raw.decode()) if isinstance(raw, (bytes, bytearray)) else json.loads(raw)
        )
        return ticket_id, payload

    async def status(self, ticket_id: str) -> dict[str, tp.Any]:
        """Получение статуса задачи"""
        hkey = f"{self.tprefix}{ticket_id}"
        data = await self.redis.hgetall(hkey)
        if not data:
            return {"state": "not_found"}
        data = {
            k.decode(): (v.decode() if isinstance(v, (bytes, bytearray)) else v)
            for k, v in data.items()
        }

        q_list = await self.redis.lrange(self.qkey, 0, -1)
        try:
            pos = q_list.index(ticket_id.encode())
        except ValueError:
            pos = 0
        data["approx_position"] = pos
        return data

    async def dequeue_blocking(self, timeout: int = 0) -> tuple[str, dict[str, tp.Any]] | None:
        """Атомарно: BRPOPLPUSH main -> processing и возврат payload"""
        raw_tid = await self.redis.brpoplpush(self.qkey, self.pkey, timeout=timeout)
        if not raw_tid:
            return None

        ticket_id = raw_tid.decode() if isinstance(raw_tid, (bytes, bytearray)) else str(raw_tid)

        hkey = f"{self.tprefix}{ticket_id}"
        now = int(time.time())
        await self.redis.hset(hkey, mapping={"updated_at": now}) # добавил обновление по времени из main -> processing

        raw = await self.redis.hget(hkey, "payload")
        if raw is None:
            return ticket_id, {}
        payload = (
            json.loads(raw.decode()) if isinstance(raw, (bytes, bytearray)) else json.loads(raw)
        )
        return ticket_id, payload

    async def ack(self, ticket_id: str) -> None:
        """Подтверждение обработки: удаляем из processing."""
        await self.redis.lrem(self.pkey, 1, ticket_id)

    async def sweep_processing(self, stale_sec: int = 60) -> int:
        """
        Возвращает в основную очередь тикеты, застрявшие в processing дольше stale_sec
        Если хэш исчез — просто удаляем из processing.
        """
        now = int(time.time())
        ids = await self.redis.lrange(self.pkey, 0, -1)
        requeued = 0

        for raw in ids:
            ticket_id = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
            hkey = f"{self.tprefix}{ticket_id}"
            data = await self.redis.hgetall(hkey)

            if not data:
                # Хэш пропал/протух — очистим processing
                await self.redis.lrem(self.pkey, 1, ticket_id)
                continue

            # Вспомогательный декодер
            def _get(k: bytes, default: str = "") -> str:
                v = data.get(k)
                return v.decode() if isinstance(v, (bytes, bytearray)) else (v or default)

            state = _get(b"state")
            try:
                updated_at = int(_get(b"updated_at", "0"))
            except ValueError:
                updated_at = 0

            # Реальные «подвисшие» состояния — queued/running без движения
            if state in {"queued", "running"} and now - updated_at >= stale_sec:
                await self.requeue(ticket_id, reason="sweep: stale in processing")
                requeued += 1

        return requeued
