from httpx import AsyncClient


class TestBaseAPI:
    """Тест базовых эндпоинтов"""

    async def test_healthcheck(self, client: AsyncClient) -> None:
        """Тест жизнеспособности сервиса"""
        res = await client.get("/healthcheck")
        assert res.status_code == 200

    async def test_version(self, client: AsyncClient) -> None:
        """Тест эндпоинта получения версии"""
        res = await client.get("/version")

        assert res.status_code == 200
