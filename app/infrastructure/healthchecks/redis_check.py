import typing as tp

from app.common.storages.redis import RedisStorage
from app.infrastructure.healthchecks.interfaces import IHealthCheck


class RedisHealthCheck(IHealthCheck):
    def __init__(self, redis_storage: RedisStorage):
        self.redis_storage = redis_storage

    @property
    def name(self) -> str:
        return "redis"

    async def check(self) -> dict[str, tp.Any]:
        try:
            await self.redis_storage.client.ping()
            return {"status": "ok", "message": "Connection successful"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
