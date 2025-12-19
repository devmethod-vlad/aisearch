import typing as tp
from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class OrderDirection(str, Enum):
    """Направление сортировки"""

    ASC = "asc"
    DESC = "desc"


class Condition(str, Enum):
    """И или Или"""

    AND = "and"
    OR = "or"


class OrderingFilter(BaseModel):
    """Фильтр для сортировки."""

    field: str
    direction: OrderDirection = OrderDirection.ASC


class PaginationFilter(BaseModel):
    """Фильтр для пагинации."""

    limit: int | None = None
    offset: int | None = None


class NestedFilter(BaseModel):
    """Вложенный фильтр для поддержки сложных условий."""

    condition: Condition | None = None
    filters: list["BaseFilter | NestedFilter"]

    def to_dict(self) -> dict[str, tp.Any]:
        """Возвращает словарь с полями фильтра, исключая None значения."""
        return self.model_dump(exclude_none=True)


class BaseFilter(BaseModel):
    """Базовый класс для всех фильтров."""

    condition: Condition | None = None
    nested_filters: list["NestedFilter | BaseFilter"] | None = None
    pagination: PaginationFilter | None = None
    ordering: list[OrderingFilter] | None = None

    model_config = ConfigDict(extra="forbid")

    def to_dict(self) -> dict[str, tp.Any]:
        """Возвращает словарь с полями фильтра, исключая None значения."""
        return self.model_dump(exclude_none=True)


class StringFilter(BaseFilter):
    """Фильтр для строковых полей."""

    eq: str | None = None
    like: str | None = None
    ilike: str | None = None
    startswith: str | None = None
    endswith: str | None = None
    in_: list[str] | None = None

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=lambda x: "in" if x == "in_" else x,
    )


class UUIDFilter(BaseFilter):
    """Фильтр для полей типа UUID."""

    eq: UUID | None = None
    in_: list[UUID] | None = None

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=lambda x: "in" if x == "in_" else x,
    )


class NumberFilter(BaseFilter):
    """Фильтр для числовых полей."""

    eq: int | None = None
    gt: int | None = None
    lt: int | None = None
    ge: int | None = None
    le: int | None = None
    between: list[int] | None = None
    in_: list[int] | None = None

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=lambda x: "in" if x == "in_" else x,
    )


class DateFilter(BaseFilter):
    """Фильтр для полей с датами."""

    eq: datetime | None = None
    gt: datetime | None = None
    lt: datetime | None = None
    ge: datetime | None = None
    le: datetime | None = None
    between: list[datetime] | None = None
    in_: list[datetime] | None = None

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=lambda x: "in" if x == "in_" else x,
    )


class BooleanFilter(BaseFilter):
    """Фильтр для булевых полей."""

    eq: bool | None = None


class JSONBFilter(BaseFilter):
    """Фильтр для полей типа JSONB."""

    contains: dict[str, tp.Any] | None = None
    has_key: str | None = None
    key_eq: dict[str, tp.Any] | None = None
    key_in: dict[str, list[tp.Any]] | None = None
