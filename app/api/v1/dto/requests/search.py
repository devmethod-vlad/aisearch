from pydantic import BaseModel


class SearchRequest(BaseModel):
    """ДТО поискового запроса"""

    query: str
    top_k: int = 5
    role: list[str] | None = None
    product: list[str] | None = None
