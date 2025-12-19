import uuid
from datetime import UTC, datetime

from pydantic import Field

from app.common.arbitrary_model import ArbitraryModel
from app.infrastructure.models.enums import RatingEnum


class KnowledgeFeedbackCreateDTO(ArbitraryModel):
    """Схема создания оценки знания"""

    id: uuid.UUID
    search_request_id: uuid.UUID
    question_id: str = ""
    rating: RatingEnum
    source: str = ""
    comment: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    modified_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class KnowledgeFeedbackSchema(KnowledgeFeedbackCreateDTO):
    """Выходная схема оценки знания"""

    pass
