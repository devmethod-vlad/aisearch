import asyncio
import io
import traceback
import typing as tp
from typing import Any

import httpx

from app.common.exceptions.exceptions import NotFoundError, RequestException
from app.common.logger import AISearchLogger
from app.infrastructure.adapters.interfaces import IEduAdapter
from app.settings.config import Settings


class EduAdapter(IEduAdapter):
    """Адаптер edu"""

    def __init__(self, settings: Settings, logger: AISearchLogger):
        self.__edu_url = settings.extract_edu.edu_emias_url
        self.__token = settings.extract_edu.edu_emias_token
        self.__page_id = settings.extract_edu.edu_emias_attachments_page_id
        self.vio_file_name = settings.extract_edu.vio_base_file_name
        self.kb_file_name = settings.extract_edu.knowledge_base_file_name
        self.__harvester_base_api_url = settings.extract_edu.base_harvester_api_url
        self.logger = logger
        self.kb_suffix = settings.extract_edu.kb_harvester_suffix
        self.vio_suffix = settings.extract_edu.vio_harvester_suffix
        self.__timeout = settings.extract_edu.edu_timeout
        self._max_retries = settings.extract_edu.deduplicated_excel_max_retries
        self._backoff_base = (
            settings.extract_edu.deduplicated_excel_retry_backoff_base_seconds
        )
        self._backoff_max = (
            settings.extract_edu.deduplicated_excel_retry_backoff_max_seconds
        )

    @property
    def edu_url(self) -> str:
        """URL EDU"""
        return self.__edu_url

    @property
    def token(self) -> str:
        """Токен EDU"""
        return self.__token

    @property
    def page_id(self) -> str:
        """ID страницы вложений EDU"""
        return self.__page_id

    @property
    def harvester_api_url(self) -> str:
        """URL data harvester"""
        return self.__harvester_base_api_url

    @property
    def timeout(self) -> int:
        """Timeout обращения к EDU"""
        return self.__timeout

    @property
    def _confluence_headers(self) -> dict[str, str]:
        """Базовые заголовки для Confluence Data Center."""
        return {
            "Authorization": f"Bearer {self.token}",
            "X-Atlassian-Token": "nocheck",
        }

    async def _request_with_retries(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Выполняет HTTP-запрос с retry/backoff для Confluence API."""
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                response = await client.request(method, url, **kwargs)

                if response.status_code == 429 or 500 <= response.status_code < 600:
                    message = (
                        f"Confluence request failed with retryable status "
                        f"{response.status_code} for {url}"
                    )
                    self.logger.warning(
                        f"Попытка {attempt}/{self._max_retries} для {method} {url} "
                        f"вернула статус {response.status_code}"
                    )
                    if attempt == self._max_retries:
                        raise RequestException(message)

                    backoff_seconds = min(
                        self._backoff_max,
                        self._backoff_base * (2 ** (attempt - 1)),
                    )
                    await asyncio.sleep(backoff_seconds)
                    continue

                if 400 <= response.status_code < 500:
                    response.raise_for_status()

                return response
            except (
                httpx.ConnectError,
                httpx.TimeoutException,
                httpx.TransportError,
            ) as error:
                last_error = error
                self.logger.warning(
                    f"Попытка {attempt}/{self._max_retries} для {method} {url} "
                    f"завершилась ошибкой {type(error).__name__}: {error}"
                )
                if attempt == self._max_retries:
                    break

                backoff_seconds = min(
                    self._backoff_max,
                    self._backoff_base * (2 ** (attempt - 1)),
                )
                await asyncio.sleep(backoff_seconds)

        if last_error is not None:
            raise RequestException(
                f"({type(last_error)}): {last_error}"
            ) from last_error

        raise RequestException(f"Не удалось выполнить запрос {method} {url}")

    async def _fetch_attachments(
        self,
        client: httpx.AsyncClient,
        page_id: str,
        filename: str | None = None,
        expand: str | None = None,
    ) -> list[dict[str, tp.Any]]:
        """Возвращает все attachment со страницы с поддержкой пагинации."""
        attachments: list[dict[str, tp.Any]] = []
        start = 0
        limit = 100

        while True:
            params: dict[str, tp.Any] = {"start": start, "limit": limit}
            if filename:
                params["filename"] = filename
            if expand:
                params["expand"] = expand

            url = f"{self.edu_url}/rest/api/content/{page_id}/child/attachment"
            response = await self._request_with_retries(
                client,
                method="GET",
                url=url,
                headers=self._confluence_headers,
                params=params,
            )
            data = response.json()
            batch = data.get("results", [])
            attachments.extend(batch)

            size = data.get("size", len(batch))
            if size == 0:
                break

            if "_links" in data and "next" in data["_links"]:
                start += size
                continue

            if len(batch) < limit:
                break

            start += len(batch)

        if filename:
            attachments = [item for item in attachments if item.get("title") == filename]

        return attachments

    async def _find_attachment_by_filename(
        self,
        client: httpx.AsyncClient,
        filename: str,
        page_id: str,
    ) -> dict[str, tp.Any] | None:
        """Ищет attachment по имени на указанной странице."""
        attachments = await self._fetch_attachments(
            client=client,
            page_id=page_id,
            filename=filename,
            expand="version",
        )
        return attachments[0] if attachments else None

    async def get_attachment_id_from_edu(
        self, filename: str, page_id: str | None = None
    ) -> str:
        """Получение id вложения на EDU"""
        page_id = page_id or self.page_id
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            attachment = await self._find_attachment_by_filename(
                client=client,
                filename=filename,
                page_id=page_id,
            )

        if not attachment:
            raise NotFoundError

        return attachment["id"]

    async def _download_file_from_edu(
        self, filename: str, page_id: str | None = None
    ) -> io.BytesIO:
        """Скачать файл-вложение по названию через download-ссылку"""
        page_id = page_id or self.page_id

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            attachment = await self._find_attachment_by_filename(
                client=client,
                filename=filename,
                page_id=page_id,
            )
            if not attachment:
                raise NotFoundError(
                    f"Файл '{filename}' не найден на странице {page_id}"
                )

            download_link = attachment["_links"].get("download")
            if not download_link.startswith("http"):
                download_link = f"{self.edu_url}{download_link}"

            file_resp = await self._request_with_retries(
                client,
                method="GET",
                url=download_link,
                headers=self._confluence_headers,
            )

            file_data = io.BytesIO(file_resp.content)
            self.logger.info(
                f"Файл '{filename}' скачан по ссылке {download_link} "
                f"({len(file_resp.content)} байт)"
            )
            return file_data

    async def _get_attachment_versions(
            self,
            client: httpx.AsyncClient,
            attachment_id: str,
    ) -> list[dict[str, tp.Any]]:
        """Получает все версии attachment с поддержкой пагинации.

        Если Confluence Data Center не возвращает историю версий attachment по
        endpoint `/rest/api/content/{attachment_id}/version`, метод логирует
        предупреждение и возвращает пустой список. Это позволяет не считать
        успешную загрузку файла ошибочной только из-за недоступной очистки версий.
        """
        versions: list[dict[str, tp.Any]] = []
        start = 0
        limit = 100

        while True:
            url = f"{self.edu_url}/rest/api/content/{attachment_id}/version"

            try:
                response = await self._request_with_retries(
                    client,
                    method="GET",
                    url=url,
                    headers=self._confluence_headers,
                    params={"start": start, "limit": limit},
                )
            except httpx.HTTPStatusError as error:
                if error.response.status_code == 404:
                    self.logger.warning(
                        "Confluence не вернул историю версий attachment "
                        f"{attachment_id}: endpoint {url} ответил 404. "
                        "Очистка старых версий будет пропущена."
                    )
                    return []
                raise

            data = response.json()
            batch = data.get("results", [])
            versions.extend(batch)

            size = data.get("size", len(batch))
            if size == 0:
                break

            if "_links" in data and "next" in data["_links"]:
                start += size
                continue

            if len(batch) < limit:
                break

            start += len(batch)

        return versions

    async def _prune_attachment_versions(
        self,
        client: httpx.AsyncClient,
        attachment_id: str,
        keep_last_versions: int,
    ) -> None:
        """Удаляет старые версии attachment, оставляя только последние N."""
        versions = await self._get_attachment_versions(client, attachment_id)
        self.logger.info(
            f"Найдено версий для attachment {attachment_id}: {len(versions)}"
        )

        if len(versions) <= keep_last_versions:
            return

        sorted_versions = sorted(
            versions,
            key=lambda item: int(item.get("number", 0)),
        )
        to_delete = sorted_versions[: len(sorted_versions) - keep_last_versions]

        for version in to_delete:
            version_number = int(version.get("number", 0))
            if version_number <= 0:
                continue

            try:
                delete_url = (
                    f"{self.edu_url}/rest/api/content/{attachment_id}/version/"
                    f"{version_number}"
                )
                await self._request_with_retries(
                    client,
                    method="DELETE",
                    url=delete_url,
                    headers=self._confluence_headers,
                )
                self.logger.info(
                    f"Удалена старая версия attachment {attachment_id}: {version_number}"
                )
            except Exception:
                self.logger.exception(
                    f"Не удалось удалить версию {version_number} "
                    f"attachment {attachment_id}"
                )

    async def upload_attachment_to_edu(
        self,
        filename: str,
        content: bytes,
        content_type: str,
        page_id: str | None = None,
    ) -> dict:
        """Загружает файл как attachment на страницу EDU/Confluence."""
        page_id = page_id or self.page_id
        url = f"{self.edu_url}/rest/api/content/{page_id}/child/attachment"

        files = {"file": (filename, content, content_type)}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await self._request_with_retries(
                client,
                method="POST",
                url=url,
                headers=self._confluence_headers,
                files=files,
            )

        response.raise_for_status()
        return response.json()

    async def upload_or_update_attachment_to_edu(
            self,
            filename: str,
            content: bytes,
            content_type: str,
            keep_last_versions: int,
            page_id: str | None = None,
    ) -> None:
        """Загружает новый attachment или новую версию существующего файла.

        Если attachment с таким filename уже есть на странице, создается новая
        версия файла. После успешной загрузки старые версии удаляются так, чтобы
        осталось не больше keep_last_versions последних версий.

        Очистка старых версий выполняется best-effort: если Confluence не отдает
        историю версий или удаление старых версий недоступно, сам факт успешной
        загрузки файла не считается ошибкой.
        """
        page_id = page_id or self.page_id
        files = {"file": (filename, content, content_type)}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            existing_attachment = await self._find_attachment_by_filename(
                client=client,
                filename=filename,
                page_id=page_id,
            )

            should_prune_versions = False

            if existing_attachment:
                attachment_id = existing_attachment["id"]
                upload_url = (
                    f"{self.edu_url}/rest/api/content/{page_id}/child/attachment/"
                    f"{attachment_id}/data"
                )
                await self._request_with_retries(
                    client,
                    method="POST",
                    url=upload_url,
                    headers=self._confluence_headers,
                    files=files,
                )
                should_prune_versions = True

                self.logger.info(
                    f"Создана новая версия attachment '{filename}' (id={attachment_id})"
                )
            else:
                upload_url = (
                    f"{self.edu_url}/rest/api/content/{page_id}/child/attachment"
                )
                response = await self._request_with_retries(
                    client,
                    method="POST",
                    url=upload_url,
                    headers=self._confluence_headers,
                    files=files,
                )
                payload = response.json()
                attachment_id = payload["results"][0]["id"]

                self.logger.info(
                    f"Создан новый attachment '{filename}' (id={attachment_id})"
                )

            if should_prune_versions:
                try:
                    await self._prune_attachment_versions(
                        client=client,
                        attachment_id=attachment_id,
                        keep_last_versions=keep_last_versions,
                    )
                except Exception:
                    self.logger.exception(
                        "Файл успешно загружен, но не удалось очистить старые "
                        f"версии attachment '{filename}' (id={attachment_id})"
                    )

    async def download_vio_base_file(self) -> io.BytesIO:
        """Скачать файл базы ВИО с EDU"""
        return await self._download_file_from_edu(self.vio_file_name)

    async def download_kb_base_file(self) -> io.BytesIO:
        """Скачать файл базы знаний с EDU"""
        return await self._download_file_from_edu(self.kb_file_name)

    async def provoke_harvest_to_edu(self, harvest_type: str = "all") -> bool:
        """Вызывает обновление файлов на EDU по типу ('vio', 'kb' или 'all').
        При 'all' выполняет оба запроса последовательно.
        Возвращает True, если все запросы завершились успешно, иначе False.
        """
        harvest_kb = f"{self.harvester_api_url}{self.kb_suffix}"
        harvest_vio = f"{self.harvester_api_url}{self.vio_suffix}"

        self.logger.info("🐢 Запуск сбора данных на EDU...")

        tasks = []

        if harvest_type in ("kb", "all"):
            tasks.append(("kb", harvest_kb))
        if harvest_type in ("vio", "all"):
            tasks.append(("vio", harvest_vio))

        if not tasks:
            self.logger.warning(f"⚠️ Неизвестный тип сбора данных: {harvest_type!r}")
            return False

        results = []

        async with httpx.AsyncClient(timeout=120) as client:
            for name, url in tasks:
                try:
                    response = await client.get(url)

                    if response.status_code == 200:
                        self.logger.info(f"✅ Сбор данных '{name}' успешно завершён")
                        results.append(True)
                    else:
                        self.logger.warning(
                            f"⚠️ Сбор данных '{name}' вернул статус {response.status_code}, "
                            f"тело: {response.text[:200]}"
                        )
                        results.append(False)

                except (httpx.ConnectError, httpx.TimeoutException) as e:
                    self.logger.error(
                        f"❌ Ошибка при сборе данных '{name}' ({type(e)}): {traceback.format_exc()}"
                    )
                    results.append(False)

        success = all(results)

        if len(tasks) > 1 and not success:
            successful_count = sum(1 for r in results if r)
            self.logger.warning(
                f"⚠️ Удалось обновить {successful_count}/{len(results)} источников"
            )

        return success
