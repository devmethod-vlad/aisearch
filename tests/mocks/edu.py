import io
from pathlib import Path

from app.infrastructure.adapters.interfaces import IEduAdapter


class MockEduAdapter(IEduAdapter):
    """Mock адаптер для EDU, который возвращает локальные файлы"""

    def __init__(self):
        self._kb_file_path = Path(__file__).parent.parent / "mocks" / "kb_from_edu.xlsx"
        self._vio_file_path = (
            Path(__file__).parent.parent / "mocks" / "vio_from_edu.xlsx"
        )

        if not self._kb_file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {self._kb_file_path}")
        if not self._vio_file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {self._vio_file_path}")

    async def get_attachment_id_from_edu(
        self, filename: str, page_id: str | None = None
    ) -> str:
        """Не используется в UpdaterService"""
        return "mock_attachment_id"

    async def _download_file_from_edu(
        self, filename: str, page_id: str | None = None
    ) -> io.BytesIO:
        """Не используется напрямую в UpdaterService"""
        if filename == "kb_base_file.xlsx":
            return io.BytesIO(self._kb_file_path.read_bytes())
        elif filename == "vio_base_file.xlsx":
            return io.BytesIO(self._vio_file_path.read_bytes())
        else:
            raise FileNotFoundError(f"Файл {filename} не найден в моках")

    async def download_vio_base_file(self) -> io.BytesIO:
        """Скачать файл базы ВИО с EDU"""
        return io.BytesIO(self._vio_file_path.read_bytes())

    async def download_kb_base_file(self) -> io.BytesIO:
        """Скачать файл базы знаний с EDU"""
        return io.BytesIO(self._kb_file_path.read_bytes())

    async def provoke_harvest_to_edu(self, harvest_type: str = "all") -> bool:
        """Не используется в UpdaterService"""
        return True
