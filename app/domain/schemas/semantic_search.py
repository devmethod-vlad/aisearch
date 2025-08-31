from pydantic import BaseModel


class SemanticSearchFoundDTO(BaseModel):
    """ДТО результата поиска"""

    id: int
    distance: float
