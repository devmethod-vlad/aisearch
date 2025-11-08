import asyncio
import io

import httpx

from app.common.exceptions.exceptions import NotFoundError, RequestError
from app.common.logger import AISearchLogger, LoggerType
from app.infrastructure.adapters.interfaces import IEduAdapter
from app.settings.config import Settings, settings



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

    async def get_attachment_id_from_edu(self, filename: str, page_id: str | None = None) -> str:
        """Получение id вложения на EDU"""
        page_id = page_id or self.page_id
        url = f"{self.edu_url}/rest/api/content/{page_id}/child/attachment"
        headers = {"Authorization": f"Bearer {self.token}"}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers)
                if response.status_code == 200:
                    data = response.json()

                    for attachment in data.get("results", []):
                        if attachment.get("title") == filename:
                            return attachment["id"]
                    raise NotFoundError
                else:
                    error_text = f"EDU вернул статус - {response.status_code}"
                    self.logger.warning(error_text)
                    raise RequestError(error_text)
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                raise RequestError(str(e))

    async def _download_file_from_edu(self, filename: str, page_id: str | None = None) -> io.BytesIO:
        """Скачать файл-вложение по названию через download-ссылку"""
        page_id = page_id or self.page_id
        url = f"{self.edu_url}/rest/api/content/{page_id}/child/attachment"
        headers = {"Authorization": f"Bearer {self.token}"}

        async with httpx.AsyncClient() as client:

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
                raise NotFoundError(f"Файл '{filename}' не найден на странице {page_id}")

            download_headers = {"Authorization": f"Bearer {self.token}"}
            file_resp = await client.get(download_link, headers=download_headers)
            file_resp.raise_for_status()

            file_data = io.BytesIO(file_resp.content)
            self.logger.info(f"Файл '{filename}' скачан по ссылке {download_link} ({len(file_resp.content)} байт)")
            return file_data

    async def download_vio_base_file(self) -> io.BytesIO:
        """Скачать файл базы ВИО с EDU"""
        return await self._download_file_from_edu(self.vio_file_name)

    async def download_kb_base_file(self) -> io.BytesIO:
        """Скачать файл базы знаний с EDU"""
        return await self._download_file_from_edu(self.kb_file_name)

    async def provoke_harvest_to_edu(self, harvest_type: str) -> bool:
        """Вызывает обновление файлов на EDU по типу ('vio' или 'kb').

        Возвращает True, если запрос завершился со статусом 200, иначе False.
        """
        harvest_kb = f"{self.harvester_api_url}/knowledge-base/collect-all"
        harvest_vio = f"{self.harvester_api_url}/vio/runtime_harvest"


        self.logger.info("🚀 Запуск сбора данных на EDU...")
        if harvest_type == "kb":
            url = harvest_kb
        elif harvest_type == "vio":
            url = harvest_vio
        else:
            self.logger.warning(f"⚠️ Неизвестный тип сбора данных: {harvest_type!r}")
            return False

        async with httpx.AsyncClient(timeout=120) as client:
            try:
                response = await client.get(url)

                if response.status_code == 200:
                    self.logger.info(f"✅ Сбор данных '{harvest_type}' успешно завершён")
                    return True

                self.logger.warning(
                    f"⚠️ Сбор данных '{harvest_type}' вернул статус {response.status_code}, тело: {response.text[:200]}"
                )
                return False

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                self.logger.error(f"❌ Ошибка при выполнении сбора данных '{harvest_type}': {e}")
                return False

