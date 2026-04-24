import uuid
from datetime import UTC, datetime

from pydantic import Field

from app.common.arbitrary_model import ArbitraryModel


class GlossaryElementCreateDTO(ArbitraryModel):
    """DTO для массовой записи элемента глоссария в БД после загрузки из API."""

    id: uuid.UUID
    abbreviation: str = ""
    term: str = ""
    definition: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    modified_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class GlossaryElementSchema(GlossaryElementCreateDTO):
    """Схема чтения элемента глоссария из репозитория."""

    pass
