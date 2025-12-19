import typing as tp

from sqlalchemy import text

from app.common.database import Database
from app.infrastructure.healthchecks.interfaces import IHealthCheck


class PostgresHealthCheck(IHealthCheck):
    def __init__(self, database: Database):
        self.database = database

    @property
    def name(self) -> str:
        return "postgres"

    async def check(self) -> dict[str, tp.Any]:
        try:
            async with self.database.session_factory() as session:
                await session.execute(text("SELECT 1"))
                await session.commit()
                return {"status": "ok", "message": "Connection successful"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
