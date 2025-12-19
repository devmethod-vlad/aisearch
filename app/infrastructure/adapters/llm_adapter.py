import typing as tp

import httpx

from app.common.logger import AISearchLogger
from app.infrastructure.adapters.interfaces import IVLLMAdapter
from app.settings.config import Settings


class VLLMAdapter(IVLLMAdapter):
    """Асинхронный клиент к локальному vLLM серверу (OpenAI-compatible /v1)."""

    def __init__(self, settings: Settings, logger: AISearchLogger):
        self.settings = settings
        self.log = logger
        self._client = httpx.AsyncClient(
            base_url=self.settings.vllm.base_url,
            headers=self._headers(),
            timeout=self.settings.vllm.request_timeout,
        )

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.settings.vllm.api_key:
            h["Authorization"] = f"Bearer {self.settings.vllm.api_key}"
        return h

    async def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra: dict[str, tp.Any] | None = None,
    ) -> str:
        """Асинхронное обращение к модели"""
        payload: dict[str, tp.Any] = {
            "model": self.settings.vllm.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens or self.settings.vllm.max_output_tokens,
            "temperature": (
                temperature
                if temperature is not None
                else self.settings.vllm.temperature
            ),
            "top_p": top_p if top_p is not None else self.settings.vllm.top_p,
            "stream": self.settings.vllm.stream,
        }
        if extra:
            payload.update(extra)
        try:
            r = await self._client.post("/chat/completions", json=payload)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            self.log.error(
                f"vLLM request failed: {e.response.status_code} {e.response.text}"
            )
            raise
        data = r.json()
        return data["choices"][0]["message"]["content"]

    async def close(self) -> None:
        """Закрытие подключения"""
        await self._client.aclose()
