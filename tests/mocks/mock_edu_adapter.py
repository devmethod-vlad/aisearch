from io import BytesIO

import aiofiles

from app.domain.adapters.interfaces import IEduAdapter


class MockEduAdapter(IEduAdapter):
    """Мок адаптера EDU"""

    async def create_or_update_file_on_edu(self, filename: str, file_data: BytesIO) -> None:
        """Создание/обновление файла на EDU"""
        async with aiofiles.open(f"tests/tmp/knowledge_base_test_dir/{filename}", "wb") as f:
            await f.write(file_data.read())

    async def get_attachment_id_from_edu(self, filename: str) -> str:
        """Получение id вложения на EDU"""
        return 1000
