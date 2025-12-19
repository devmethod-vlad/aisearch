import typing as tp

import httpx

from app.infrastructure.healthchecks.interfaces import IHealthCheck
from app.settings.config import settings


class OpenSearchHealthCheck(IHealthCheck):
    @property
    def name(self) -> str:
        return "opensearch"

    async def check(self) -> dict[str, tp.Any]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"http://{settings.opensearch.host}:{settings.opensearch.port}/_cluster/health"
                )
                if response.status_code == 200:
                    data: dict[str, tp.Any] = response.json()
                    return {
                        "status": "ok",
                        "message": f"Cluster status: {data.get('status', 'unknown')}",
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"HTTP {response.status_code}",
                    }
        except Exception as e:
            return {"status": "error", "message": str(e)}
