import typing as tp

import httpx

from app.infrastructure.healthchecks.interfaces import IHealthCheck
from app.settings.config import settings


class MilvusHealthCheck(IHealthCheck):
    @property
    def name(self) -> str:
        return "milvus"

    async def check(self) -> dict[str, tp.Any]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"http://{settings.milvus.host}:{settings.milvus.web_ui_port}/healthz"
                )
                if response.status_code == 200:
                    return {"status": "ok", "message": "Service is healthy"}
                else:
                    return {
                        "status": "error",
                        "message": f"HTTP {response.status_code}",
                    }
        except Exception as e:
            return {"status": "error", "message": str(e)}
