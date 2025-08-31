from pydantic import BaseModel


class CollectDataInfoResponse(BaseModel):
    """Информация о сборе данных БЗ со страницы"""

    page_id: str
    message: str


class CollectDataResponse(BaseModel):
    """Ответ на сбор данных БЗ"""

    parsing_error: bool = False
    pages_info: list[CollectDataInfoResponse] = []
    message: str = ""
