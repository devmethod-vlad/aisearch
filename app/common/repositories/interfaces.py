import typing as tp
from abc import ABC, abstractmethod
from collections.abc import Iterable

from pydantic import BaseModel
from sqlalchemy.orm import DeclarativeBase

from app.common.filters.filters import BaseFilter

TModel = tp.TypeVar("TModel", bound=DeclarativeBase)
TCreateDto = tp.TypeVar("TCreateDto", bound=BaseModel)
TUpdateDto = tp.TypeVar("TUpdateDto", bound=BaseModel)
TResponseDto = tp.TypeVar("TResponseDto", bound=BaseModel)


class IRepository(ABC, tp.Generic[TModel, TCreateDto, TUpdateDto, TResponseDto]):
    """Абстрактный CRUD - репозиторий"""

    @abstractmethod
    async def create(
        self,
        create_dto: TCreateDto,
        response_dto: type[TResponseDto] | None = None,
    ) -> TResponseDto:
        """Создание объекта"""
        pass

    @abstractmethod
    async def bulk_create(
        self,
        bulk_create_dto: list[TCreateDto],
        response_dto: type[TResponseDto] | None = None,
        batch_size: int = 5000,
    ) -> list[TResponseDto]:
        """Создание нескольких объектов с разбиением на пакеты"""
        pass

    @abstractmethod
    async def get_one(
        self, filters: BaseFilter, response_dto: type[TResponseDto] | None = None
    ) -> TResponseDto:
        """Получение одного объекта"""
        pass

    @abstractmethod
    async def get_list(
        self,
        response_dto: type[TResponseDto] | None = None,
        filters: BaseFilter | None = None,
    ) -> list[TResponseDto]:
        """Получение списка объектов"""
        pass

    @abstractmethod
    async def update(
        self,
        update_dto: TUpdateDto,
        filters: BaseFilter,
        response_dto: type[TResponseDto] | None = None,
    ) -> TResponseDto:
        """Обновление объекта"""
        pass

    @abstractmethod
    async def delete(self, filters: BaseFilter) -> None:
        """Удаление объекта"""
        pass

    @abstractmethod
    def to_dto(
        self,
        instance: TModel | Iterable[TModel],
        dto: type[TResponseDto] | None = None,
    ) -> TResponseDto | list[TResponseDto]:
        """Преобразование моделей SQLAlchemy к DTO"""
        pass

    @abstractmethod
    async def _refresh(
        self,
        instance: TModel,
        auto_refresh: bool | None = None,
        attribute_names: Iterable[str] | None = None,
        with_for_update: bool | None = None,
    ) -> None:
        """Обновление объекта в сессии"""
        pass

    @abstractmethod
    async def _execute(self, statement: tp.Any) -> tp.Any:
        """Выполнение SQL-запроса"""
        pass

    @abstractmethod
    def _build_filter_query(self, stmt: tp.Any, filters: BaseFilter | None) -> tp.Any:
        """Создание SQL-запроса с условиями WHERE, сортировкой и пагинацией"""
        pass

    @abstractmethod
    def _build_filter_conditions(
        self,
        filters: BaseFilter | tp.Any,
        parent_condition: tp.Any = None,
    ) -> list[tp.Any]:
        """Построение условий фильтрации"""
        pass

    @abstractmethod
    def _apply_field_filter(
        self, field: tp.Any, field_filter: dict[str, tp.Any]
    ) -> list[tp.Any]:
        """Применение фильтра к конкретному полю"""
        pass

    @abstractmethod
    def _apply_jsonb_filter(
        self, field: tp.Any, jsonb_filter: dict[str, tp.Any]
    ) -> list[tp.Any]:
        """Применение JSONB-фильтра к конкретному полю"""
        pass

    @staticmethod
    @abstractmethod
    def check_not_found(item_or_none: TModel | None) -> TModel:
        """Проверка на существование объекта в базе"""
        pass

    @abstractmethod
    async def count(self, filters: BaseFilter | None = None) -> int:
        """Получение количества объектов"""
        pass

    @abstractmethod
    async def get_paginated(
        self,
        filters: BaseFilter | None = None,
        response_dto: BaseModel | None = None,
    ) -> tuple[list[BaseModel], int, bool]:
        """Получение списка объектов с информацией о пагинации (items, total, has_more)"""
