import io
import traceback

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

    async def get_attachment_id_from_edu(
        self, filename: str, page_id: str | None = None
    ) -> str:
        """Получение id вложения на EDU"""
        page_id = page_id or self.page_id
        url = f"{self.edu_url}/rest/api/content/{page_id}/child/attachment"
        headers = {"Authorization": f"Bearer {self.token}"}

        self.logger.info(f"Получаем id вложения на страницe {page_id}, URL: {url}")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(url, headers=headers)
                self.logger.info(f"Ответ EDU: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    for attachment in data.get("results", []):
                        if attachment.get("title") == filename:
                            return attachment["id"]
                    raise NotFoundError
                else:
                    error_text = f"EDU вернул статус - {response.status_code}"
                    self.logger.warning(error_text)
                    raise RequestException(error_text)
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                raise RequestException(f"({type(e)}): {traceback.format_exc()}")

    async def _download_file_from_edu(
        self, filename: str, page_id: str | None = None
    ) -> io.BytesIO:
        """Скачать файл-вложение по названию через download-ссылку"""
        page_id = page_id or self.page_id
        url = f"{self.edu_url}/rest/api/content/{page_id}/child/attachment"
        headers = {"Authorization": f"Bearer {self.token}"}

        async with httpx.AsyncClient(timeout=self.timeout) as client:

            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            for attachment in data.get("results", []):
                if attachment.get("title") == filename:
                    download_link = attachment["_links"].get("download")
                    if not download_link.startswith("http"):

                        download_link = f"{self.edu_url}{download_link}"
                    break
            else:
                raise NotFoundError(
                    f"Файл '{filename}' не найден на странице {page_id}"
                )

            download_headers = {"Authorization": f"Bearer {self.token}"}
            file_resp = await client.get(download_link, headers=download_headers)
            file_resp.raise_for_status()

            file_data = io.BytesIO(file_resp.content)
            self.logger.info(
                f"Файл '{filename}' скачан по ссылке {download_link} ({len(file_resp.content)} байт)"
            )
            return file_data

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
