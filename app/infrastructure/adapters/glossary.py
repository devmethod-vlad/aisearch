import asyncio
import uuid
from typing import Any

import httpx

from app.common.exceptions.exceptions import RequestException
from app.common.logger import AISearchLogger
from app.domain.schemas.glossary import GlossaryElementCreateDTO
from app.infrastructure.adapters.interfaces import IGlossaryAdapter
from app.settings.config import Settings


class GlossaryAdapter(IGlossaryAdapter):
    """Адаптер для постраничной загрузки глоссария аббревиатур из внешнего API."""

    def __init__(self, settings: Settings, logger: AISearchLogger):
        self.settings = settings
        self.logger = logger

    async def fetch_all(self) -> list[GlossaryElementCreateDTO]:
        """Получает и нормализует все элементы глоссария через limit/offset-пагинацию."""
        glossary_settings = self.settings.glossary
        timeout = httpx.Timeout(glossary_settings.request_timeout)
        offset = 0
        total: int | None = None
        result: list[GlossaryElementCreateDTO] = []

        async with httpx.AsyncClient(timeout=timeout) as client:
            while True:
                payload = await self._fetch_page_with_retry(client=client, offset=offset)
                data, count, page_total = self._parse_page(payload=payload, offset=offset)

                if total is None:
                    total = page_total
                    if total == 0:
                        self.logger.info("Внешний API глоссария вернул 0 элементов")
                        break

                if not data or count == 0:
                    self.logger.info(
                        "Остановка пагинации глоссария: пустая страница (offset=%s)",
                        offset,
                    )
                    break

                normalized_items = self._normalize_items(data=data, offset=offset)
                result.extend(normalized_items)

                if len(result) >= (total or 0):
                    break

                offset += len(data)

        self.logger.info(
            "Глоссарий успешно загружен из API: total=%s, fetched=%s",
            total if total is not None else 0,
            len(result),
        )
        return result

    async def _fetch_page_with_retry(
        self, client: httpx.AsyncClient, offset: int
    ) -> dict[str, Any]:
        """Выполняет запрос страницы к API с retry и экспоненциальным backoff."""
        glossary_settings = self.settings.glossary
        params = {
            "limit": glossary_settings.page_limit,
            "offset": offset,
        }
        last_error: Exception | None = None

        for attempt in range(1, glossary_settings.max_retries + 1):
            try:
                response = await client.get(glossary_settings.api_url, params=params)
                if response.status_code >= 400:
                    response.raise_for_status()
                return response.json()
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_error = exc
                if attempt >= glossary_settings.max_retries:
                    break
                delay = self._get_retry_delay(attempt)
                self.logger.warning(
                    "Ошибка соединения с API глоссария (offset=%s, attempt=%s): %s. Retry через %s сек.",
                    offset,
                    attempt,
                    type(exc).__name__,
                    delay,
                )
                await asyncio.sleep(delay)
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status_code = exc.response.status_code
                retryable = status_code == 429 or 500 <= status_code < 600
                if (not retryable) or attempt >= glossary_settings.max_retries:
                    break
                delay = self._get_retry_delay(attempt)
                self.logger.warning(
                    "HTTP ошибка API глоссария (offset=%s, attempt=%s, status=%s). Retry через %s сек.",
                    offset,
                    attempt,
                    status_code,
                    delay,
                )
                await asyncio.sleep(delay)

        raise RequestException(
            "Не удалось получить страницу глоссария "
            f"(offset={offset}) после {glossary_settings.max_retries} попыток: {last_error}"
        )

    def _parse_page(self, payload: dict[str, Any], offset: int) -> tuple[list[dict], int, int]:
        """Проверяет структуру ответа API и возвращает data/count/total."""
        if not isinstance(payload, dict):
            raise RequestException(
                f"Некорректный формат ответа API глоссария на offset={offset}: ожидается объект"
            )

        data = payload.get("data")
        count = payload.get("count")
        total = payload.get("total")

        if not isinstance(data, list) or not isinstance(count, int) or not isinstance(total, int):
            raise RequestException(
                "Некорректная структура ответа API глоссария: ожидаются поля "
                "data(list), count(int), total(int)"
            )

        data_length = len(data)
        if count != data_length:
            self.logger.warning(
                "Несовпадение count и фактической длины data (offset=%s): count=%s, len(data)=%s",
                offset,
                count,
                data_length,
            )
            count = data_length

        return data, count, total

    def _normalize_items(
        self, data: list[dict[str, Any]], offset: int
    ) -> list[GlossaryElementCreateDTO]:
        """Нормализует элементы страницы API к формату DTO для записи в БД."""
        normalized: list[GlossaryElementCreateDTO] = []

        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise RequestException(
                    f"Некорректный элемент data в API глоссария на offset={offset}, index={index}"
                )

            abbreviation = self._normalize_string(item.get("abbreviation"))
            term = self._normalize_string(item.get("term"))
            definition = self._normalize_string(item.get("definition"))

            if len(abbreviation) > 500:
                self.logger.warning(
                    "abbreviation длиннее 500 символов и будет обрезан (offset=%s, index=%s)",
                    offset,
                    index,
                )
                abbreviation = abbreviation[:500]

            normalized.append(
                GlossaryElementCreateDTO(
                    id=uuid.uuid4(),
                    abbreviation=abbreviation,
                    term=term,
                    definition=definition,
                )
            )

        return normalized

    def _get_retry_delay(self, attempt: int) -> float:
        """Возвращает длительность паузы retry по экспоненциальному backoff."""
        glossary_settings = self.settings.glossary
        raw_delay = glossary_settings.retry_backoff_base_seconds * (2 ** (attempt - 1))
        return min(raw_delay, glossary_settings.retry_backoff_max_seconds)

    @staticmethod
    def _normalize_string(value: Any) -> str:
        """Преобразует входное значение API в очищенную строку."""
        if value is None:
            return ""
        return str(value).strip()
