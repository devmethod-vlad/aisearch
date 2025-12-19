import typing as tp
from collections.abc import Iterable

from pydantic import BaseModel
from sqlalchemy import (
    BinaryExpression,
    Delete,
    Result,
    ScalarResult,
    Select,
    Update,
    ValuesBase,
    and_,
    asc,
    delete,
    desc,
    func,
    insert,
    or_,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, InstrumentedAttribute

from app.common.exceptions.exceptions import ConflictError, NotFoundError
from app.common.filters.filters import (
    BaseFilter,
    Condition,
    NestedFilter,
    OrderDirection,
)
from app.common.repositories.interfaces import IRepository


class SQLAlchemyRepository(IRepository):
    """CRUD - репозиторий для SQLAlchemy"""

    model: type[DeclarativeBase] | None = None
    response_dto: type[BaseModel] | None = None

    def __init__(
        self,
        session: AsyncSession,
    ):
        self.session = session
        self.auto_commit = None
        self.auto_refresh = None

    async def create(
        self,
        create_dto: BaseModel,
        response_dto: BaseModel | None = None,
    ) -> BaseModel:
        """Создание объекта"""
        stmt = (
            insert(self.model).values(**create_dto.model_dump()).returning(self.model)
        )

        res = await self._execute(stmt)

        return self.to_dto(res.scalar_one(), response_dto)

    async def bulk_create(
        self,
        bulk_create_dto: list[BaseModel],
        response_dto: BaseModel | None = None,
        batch_size: int = 5000,
    ) -> list[BaseModel]:
        """Создание нескольких объектов с разбиением на пакеты"""
        results = []
        for i in range(0, len(bulk_create_dto), batch_size):
            batch = bulk_create_dto[i : i + batch_size]
            stmt = (
                insert(self.model)
                .values([entity.model_dump() for entity in batch])
                .returning(self.model)
            )
            res = await self._execute(stmt)
            results.extend(res.scalars())
        return self.to_dto(results, response_dto)

    async def get_one(
        self, filters: BaseFilter, response_dto: BaseModel | None = None
    ) -> BaseModel:
        """Получение одного объекта"""
        stmt = self._build_filter_query(stmt=select(self.model), filters=filters)
        result = await self._execute(stmt)
        instance = result.scalar_one_or_none()
        self.check_not_found(instance)
        return self.to_dto(instance, response_dto)

    async def get_list(
        self,
        response_dto: BaseModel | None = None,
        filters: BaseFilter = None,
    ) -> list[BaseModel]:
        """Получение списка объектов"""
        stmt = self._build_filter_query(stmt=select(self.model), filters=filters)

        res = await self._execute(stmt)

        return self.to_dto(res.scalars(), response_dto)

    async def update(
        self,
        update_dto: BaseModel,
        filters: BaseFilter,
        response_dto: BaseModel | None = None,
    ) -> BaseModel:
        """Обновление объекта"""
        stmt = self._build_filter_query(
            stmt=update(self.model).values(**update_dto.model_dump(exclude_unset=True)),
            filters=filters,
        ).returning(self.model)
        res = (await self._execute(stmt)).scalar_one_or_none()
        self.check_not_found(res)
        return self.to_dto(res, response_dto)

    def _build_filter_query(
        self, stmt: Select | Update | Delete, filters: BaseFilter | None
    ) -> Select | Update | Delete:
        """Создает SQL-запрос с условиями WHERE, сортировкой и пагинацией."""
        if filters is None:
            return stmt

        conditions = self._build_filter_conditions(filters)

        if conditions:
            stmt = stmt.where(*conditions)

        if hasattr(filters, "ordering") and filters.ordering:
            for order in filters.ordering:
                field = getattr(self.model, order.field)
                if order.direction == OrderDirection.ASC:
                    stmt = stmt.order_by(asc(field))
                else:
                    stmt = stmt.order_by(desc(field))

        if hasattr(filters, "pagination") and filters.pagination:
            if filters.pagination.limit is not None:
                stmt = stmt.limit(filters.pagination.limit)
            if filters.pagination.offset is not None:
                stmt = stmt.offset(filters.pagination.offset)
        return stmt

    def _build_filter_conditions(
        self,
        filters: BaseFilter | NestedFilter,
        parent_condition: Condition = Condition.AND,
    ) -> list[BinaryExpression]:
        conditions = []

        current_condition = getattr(filters, "condition", parent_condition)

        for field_name, field_filter in filters.to_dict().items():
            if field_name in ("condition", "nested_filters", "filters"):
                continue

            if hasattr(self.model, field_name):
                field = getattr(self.model, field_name)
                if isinstance(field.type, JSONB):
                    if isinstance(field_filter, dict):
                        conditions.extend(self._apply_jsonb_filter(field, field_filter))
                elif isinstance(field_filter, dict):
                    conditions.extend(self._apply_field_filter(field, field_filter))

        if hasattr(filters, "nested_filters") and filters.nested_filters:
            nested_conditions = []
            for nested_filter in filters.nested_filters:
                sub_conditions = self._build_filter_conditions(
                    nested_filter, nested_filter.condition
                )
                if sub_conditions:
                    nested_conditions.append(and_(*sub_conditions))

            if nested_conditions:
                if current_condition == Condition.OR:
                    conditions.append(or_(*nested_conditions))
                else:
                    conditions.append(and_(*nested_conditions))

        elif hasattr(filters, "filters") and filters.filters:
            sub_conditions_list = []
            for sub_filter in filters.filters:
                sub_conditions = self._build_filter_conditions(
                    sub_filter, sub_filter.condition
                )
                if sub_conditions:
                    sub_conditions_list.append(and_(*sub_conditions))

            if sub_conditions_list:
                if current_condition == Condition.OR:
                    conditions.append(or_(*sub_conditions_list))
                else:
                    conditions.append(and_(*sub_conditions_list))

        if conditions:
            if current_condition == Condition.OR:
                return [or_(*conditions)]
            else:
                return [and_(*conditions)]

        return conditions

    def _apply_field_filter(
        self, field: InstrumentedAttribute, field_filter: dict[str, tp.Any]
    ) -> list[BinaryExpression]:
        """Применяет фильтр к конкретному полю."""
        conditions = []
        for operator, value in field_filter.items():
            if operator == "eq":
                conditions.append(field == value)
            elif operator == "like":
                conditions.append(field.like(f"%{value}%"))
            elif operator == "ilike":
                conditions.append(field.ilike(f"%{value}%"))
            elif operator == "startswith":
                conditions.append(field.like(f"{value}%"))
            elif operator == "endswith":
                conditions.append(field.like(f"%{value}"))
            elif operator == "in_":
                conditions.append(field.in_(value))
            elif operator == "gt":
                conditions.append(field > value)
            elif operator == "lt":
                conditions.append(field < value)
            elif operator == "ge":
                conditions.append(field >= value)
            elif operator == "le":
                conditions.append(field <= value)
            elif operator == "between":
                conditions.append(field.between(value[0], value[1]))
        return conditions

    def _apply_jsonb_filter(
        self, field: InstrumentedAttribute, jsonb_filter: dict[str, tp.Any]
    ) -> list[BinaryExpression]:
        """Применяет JSONB-фильтр к конкретному полю."""
        conditions = []
        for operator, value in jsonb_filter.items():
            if operator == "contains":
                conditions.append(field.contains(value))
            elif operator == "has_key":
                conditions.append(field.has_key(value))
            elif operator == "key_eq":
                for key, val in value.items():
                    conditions.append(field[key].astext == str(val))
            elif operator == "key_in":
                for key, val in value.items():
                    conditions.append(field[key].astext.in_(val))
        return conditions

    async def delete(self, filters: BaseFilter) -> None:
        """Удаление объекта"""
        exists_query = self._build_filter_query(select(1), filters=filters)
        exists_result = await self._execute(exists_query)

        if not exists_result.scalar_one_or_none():
            raise NotFoundError(
                f"По данным запроса в таблице {self.model.__tablename__} записей не найдено"
            )

        stmt = self._build_filter_query(delete(self.model), filters=filters)
        await self._execute(stmt)

    def to_dto(
        self,
        instance: DeclarativeBase | ScalarResult,
        dto: type[BaseModel] | None = None,
    ) -> BaseModel | list[BaseModel]:
        """Метод, преобразующий модели SQLAlchemy к dto."""
        if not dto:
            dto = self.response_dto
        if isinstance(instance, ScalarResult):
            return [dto.model_validate(row, from_attributes=True) for row in instance]
        if isinstance(instance, list):
            return [dto.model_validate(row, from_attributes=True) for row in instance]
        return dto.model_validate(instance, from_attributes=True)

    async def _refresh(
        self,
        instance: DeclarativeBase,
        auto_refresh: bool | None = None,
        attribute_names: Iterable[str] | None = None,
        with_for_update: bool | None = None,
    ) -> None:
        """Метод обновления объекта в сессии"""
        if auto_refresh is None:
            auto_refresh = self.auto_refresh

        return (
            await self.session.refresh(
                instance,
                attribute_names=attribute_names,
                with_for_update=with_for_update,
            )
            if auto_refresh
            else None
        )

    @staticmethod
    def check_not_found(item_or_none: DeclarativeBase | None) -> DeclarativeBase:
        """Метод проверки на существование в базе"""
        if item_or_none is None:
            msg = "No item found when one was expected"
            raise NotFoundError(msg)
        return item_or_none

    async def _execute(
        self, statement: ValuesBase | Select[tp.Any] | Delete
    ) -> Result[tp.Any]:
        """Метод выполнения запроса"""
        try:
            return await self.session.execute(statement)
        except IntegrityError as e:
            raise ConflictError(
                f"Конфликт записи значения в таблицу {self.model.__table__}: {str(e)}"
            )

    async def count(self, filters: BaseFilter | None = None) -> int:
        """Получение количества объектов"""
        stmt = select(func.count()).select_from(self.model)

        if filters:
            conditions = self._build_filter_conditions(filters)
            if conditions:
                stmt = stmt.where(*conditions)

        result = await self._execute(stmt)
        return result.scalar()

    async def get_paginated(
        self,
        filters: BaseFilter | None = None,
        response_dto: BaseModel | None = None,
    ) -> tuple[list[BaseModel], int, bool]:
        """Получение списка объектов с информацией о пагинации (items, total, has_more)"""
        total = await self.count(filters=filters)
        items = await self.get_list(filters=filters, response_dto=response_dto)

        has_more = False
        limit = None
        offset = 0

        if filters and hasattr(filters, "pagination") and filters.pagination:
            pagination = filters.pagination
            limit = pagination.limit
            offset = pagination.offset or 0

            if limit is not None:
                has_more = (offset + limit) < total

        return items, total, has_more
