from pydantic import BaseModel


class SearchRequest(BaseModel):
    """ДТО поискового запроса"""

    query: str
    top_k: int = 5
    role: str | None = None
    product: str | None = None
