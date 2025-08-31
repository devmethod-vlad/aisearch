from io import BytesIO

import aiohttp

from app.common.exceptions.exceptions import NotFoundError, RequestError
from app.common.logger import AISearchLogger
from app.domain.adapters.interfaces import IEduAdapter
from app.settings.config import AppSettings


class EduAdapter(IEduAdapter):
    """Адаптер EDU"""

    def __init__(self, app_settings: AppSettings, logger: AISearchLogger):
        self.edu_url = app_settings.edu_emias_url
        self.token = app_settings.edu_emias_token
        self.page_id = app_settings.edu_emias_attachments_page_id
        self.logger = logger

    async def create_or_update_file_on_edu(self, filename: str, file_data: BytesIO) -> None:
        """Создание/обновление файла на EDU"""
        try:
            attachment_id = await self.get_attachment_id_from_edu(filename=filename)
        except NotFoundError:
            url = f"{self.edu_url}/rest/api/content/{self.page_id}/child/attachment"
        else:
            url = f"{self.edu_url}/rest/api/content/{self.page_id}/child/attachment/{attachment_id}/data"

        headers = {"X-Atlassian-Token": "no-check", "Authorization": f"Bearer {self.token}"}

        form_data = aiohttp.FormData()
        form_data.add_field(
            "file",
            file_data.read(),
            filename=filename,
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(url, headers=headers, data=form_data) as response,
            ):
                if not 200 <= response.status < 300:
                    error_text = f"EDU вернул статус - {response.status}"
                    self.logger.warning(error_text)
                    raise RequestError(error_text)
        except (aiohttp.ClientConnectorError, TimeoutError) as e:
            raise RequestError(str(e))

    async def get_attachment_id_from_edu(self, filename: str) -> str:
        """Получение id вложения на EDU"""
        url = f"{self.edu_url}/rest/api/content/{self.page_id}/child/attachment"
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(url, headers=headers) as response,
            ):
                if response.status == 200:
                    data = await response.json()
                    for attachment in data["results"]:
                        if attachment["title"] == filename:
                            return attachment["id"]
                    raise NotFoundError
                else:
                    error_text = f"EDU вернул статус - {response.status}"
                    self.logger.warning(error_text)
                    raise RequestError(error_text)
        except (aiohttp.ClientConnectorError, TimeoutError) as e:
            raise RequestError(str(e))
