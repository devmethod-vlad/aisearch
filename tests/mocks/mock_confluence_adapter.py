import aiofiles

from app.domain.adapters.confluence import ConfluenceAdapter


class MockOverriddenConfluenceAdapter(ConfluenceAdapter):
    """Мок для адаптера Confluence"""

    async def get_html_content_from_page(self, page_id: str) -> str:
        """Получение HTML из страницы Confluence"""
        async with aiofiles.open(
            f"tests/mocks/knowledges/{page_id}.html", encoding="utf-8"
        ) as file:
            content = await file.read()
        return content
