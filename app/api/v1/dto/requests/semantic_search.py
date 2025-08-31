from pydantic import BaseModel


class ExampleRequest(BaseModel):
    """Запрос /example/"""

    top_k: int
