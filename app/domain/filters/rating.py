from app.common.filters.filters import BaseFilter
from app.domain.schemas.enums import RatingEnum


class RatingFilter(BaseFilter):
    """Фильтр для полей типа rating_enum"""

    eq: RatingEnum | None = None
    in_: list[RatingEnum] | None = None
