from pydantic import BaseModel


class ExampleResponse(BaseModel):
    """Ответ /example/"""

    id: int
    document: str
    distance: float
