from pydantic import BaseModel, ConfigDict


class ArbitraryModel(BaseModel):
    """Базовая модель для pydantic-схем, позволяющая вложенные структуры и алиасы"""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)


# при model_dump используем by_alias=True, exclude_none=True
