import uuid

from pydantic import Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from app.api.v1.dto.requests.base import PaginatedRequest, TimeIntervalRequest
from app.common.arbitrary_model import ArbitraryModel
from app.infrastructure.models.enums import RatingEnum


class SearchFeedbackCreateRequest(ArbitraryModel):
    """Запрос на создание обратной связи по поиску"""

    search_request_id: uuid.UUID
    question_id: str = Field(max_length=500)
    rating: RatingEnum
    source: str = Field(max_length=500)
    comment: str = ""

    @field_validator("question_id", "source")
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        """Валидация непустых строк"""
        if isinstance(v, str) and not v.strip():
            raise PydanticCustomError("field_is_empty", "Поле не может быть пустым")
        return v.strip() if isinstance(v, str) else v


class KnowledgeFeedbackCreateRequest(ArbitraryModel):
    """Запрос на создание оценки знания"""

    search_request_id: uuid.UUID
    question_id: str = Field(max_length=500)
    rating: RatingEnum
    source: str = Field(max_length=500)
    comment: str

    @field_validator("question_id", "source", "comment")
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        """Валидация непустых строк"""
        if isinstance(v, str) and not v.strip():
            raise PydanticCustomError("field_is_empty", "Поле не может быть пустым")
        return v.strip() if isinstance(v, str) else v


class SearchFeedbackBulkCreateRequest(ArbitraryModel):
    """Запрос на массовое создание обратной связи по поиску"""

    feedbacks: list[SearchFeedbackCreateRequest]

    @model_validator(mode="after")
    def check_duplicates(self) -> "SearchFeedbackBulkCreateRequest":
        """Проверка на дубликаты"""
        unique_feedbacks = []
        for feedback in self.feedbacks:
            key = (feedback.search_request_id, feedback.question_id)
            if key not in unique_feedbacks:
                unique_feedbacks.append(key)

        if len(unique_feedbacks) != len(self.feedbacks):
            raise PydanticCustomError(
                "search_request_id_question_id_duplicate",
                "Запрос содержит дублирующиеся связки search_request_id + question_id",
            )

        return self


class KnowledgeFeedbackBulkCreateRequest(ArbitraryModel):
    """Запрос на массовое создание оценки знания"""

    feedbacks: list[KnowledgeFeedbackCreateRequest]

    @model_validator(mode="after")
    def check_duplicates(self) -> "KnowledgeFeedbackBulkCreateRequest":
        """Проверка на дубликаты"""
        unique_feedbacks = []
        for feedback in self.feedbacks:
            key = (feedback.search_request_id, feedback.question_id)
            if key not in unique_feedbacks:
                unique_feedbacks.append(key)

        if len(unique_feedbacks) != len(self.feedbacks):
            raise PydanticCustomError(
                "search_request_id_question_id_duplicate",
                "Запрос содержит дублирующиеся связки search_request_id + question_id",
            )

        return self


class SearchRequestQueryRequest(PaginatedRequest, TimeIntervalRequest):
    """Запрос для получения search_request"""

    limit: int = Field(default=100, ge=1, le=200)


class SearchFeedbackQueryRequest(PaginatedRequest, TimeIntervalRequest):
    """Запрос для получения search_feedback"""

    limit: int = Field(default=100, ge=1, le=500)


class KnowledgeFeedbackQueryRequest(PaginatedRequest, TimeIntervalRequest):
    """Запрос для получения knowledge_feedback"""

    limit: int = Field(default=100, ge=1, le=500)
