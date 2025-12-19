import typing as tp

from app.common.arbitrary_model import ArbitraryModel

T = tp.TypeVar("T")


class PaginatedResponse(ArbitraryModel, tp.Generic[T]):
    """Базовый класс для пагинированных ответов"""

    items: list[T]
    total: int
    limit: int
    offset: int
    has_more: bool
