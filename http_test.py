import httpx
import asyncio

async def call_search():
    async with httpx.AsyncClient(base_url="http://localhost:5155/hybrid-search/", timeout=30) as client:
        resp = await client.post(
            "/search",
            json={"query": "пример запроса", "top_k": 5}
        )
        print(resp.status_code)
        print(resp.json())

asyncio.run(call_search())