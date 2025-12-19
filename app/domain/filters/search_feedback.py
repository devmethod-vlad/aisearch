from app.common.filters.filters import BaseFilter, DateFilter, StringFilter, UUIDFilter
from app.domain.filters.rating import RatingFilter


class SearchFeedbackFilter(BaseFilter):
    """Фильтр search_feedback"""

    id: UUIDFilter | None = None
    search_request_id: UUIDFilter | None = None
    question_id: StringFilter | None = None
    rating: RatingFilter | None = None
    source: StringFilter | None = None
    comment: StringFilter | None = None
    created_at: DateFilter | None = None
    modified_at: DateFilter | None = None
