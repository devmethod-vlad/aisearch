from app.common.logger import AISearchLogger
from app.infrastructure.adapters.interfaces import IGlossaryAdapter
from app.infrastructure.unit_of_work.interfaces import IUnitOfWork
from app.services.interfaces import IGlossaryService


class GlossaryService(IGlossaryService):
    """Сервис, который синхронизирует глоссарий из внешнего API в PostgreSQL."""

    def __init__(
        self,
        uow: IUnitOfWork,
        glossary_adapter: IGlossaryAdapter,
        logger: AISearchLogger,
    ):
        self.uow = uow
        self.glossary_adapter = glossary_adapter
        self.logger = logger

    async def sync_glossary(self) -> int:
        """Загружает глоссарий из API и атомарно перезаписывает таблицу glossary_element."""
        self.logger.info("Начало синхронизации глоссария")
        try:
            elements = await self.glossary_adapter.fetch_all()

            async with self.uow:
                count = await self.uow.glossary.replace_all(elements=elements)
                await self.uow.commit()

            self.logger.info(
                "Синхронизация глоссария завершена успешно, элементов: %s", count
            )
            return count
        except Exception:
            self.logger.exception("Ошибка синхронизации глоссария")
            raise
