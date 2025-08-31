from pathlib import Path

import pandas as pd
from _pytest.legacypath import TempdirFactory

from app.services.interfaces import IKnowledgeBaseService


class TestKnowledgeBaseService:
    """Тесты сервиса базы знаний"""

    async def test_collect_data_from_confluence_by_page_id(
        self, knowledge_base_service: IKnowledgeBaseService
    ) -> None:
        """Тест сбора данных БЗ из страницы Confluence"""
        result = await knowledge_base_service.collect_data_from_confluence_by_page_id(
            page_id="223777419"
        )
        assert result.model_dump() == {
            "parsing_error": True,
            "pages_info": [
                {"page_id": "223777419", "message": "Данные со страницы 223777419 собраны"}
            ],
            "message": "Данные собраны",
        }

    async def test_collect_all_data_from_confluence(
        self, knowledge_base_service: IKnowledgeBaseService, tmpdir_factory: TempdirFactory
    ) -> None:
        """Тест сбора всех данных БЗ"""
        path = Path(tmpdir_factory.mktemp("knowledge_base_test_dir", numbered=False))

        await knowledge_base_service.collect_all_data_from_confluence()
        df_errors = pd.read_excel(path / "ERRORS.xlsx", engine="openpyxl")
        df_correct_errors = pd.read_excel(
            "tests/mocks/knowledges/CORRECT_ERRORS.xlsx", engine="openpyxl"
        )

        df_errors = df_errors.groupby(list(df_errors.columns)).size().reset_index(name="count")
        df_correct_errors = (
            df_correct_errors.groupby(list(df_correct_errors.columns))
            .size()
            .reset_index(name="count")
        )

        pd.testing.assert_frame_equal(
            df_errors.sort_values(by=list(df_errors.columns)).reset_index(drop=True),
            df_correct_errors.sort_values(by=list(df_correct_errors.columns)).reset_index(
                drop=True
            ),
        )
