from sqlalchemy import insert, text

from app.common.exceptions.exceptions import ConflictError
from app.common.repositories.repository import SQLAlchemyRepository
from app.domain.repositories.interfaces import IGlossaryRepository
from app.domain.schemas.glossary import GlossaryElementCreateDTO, GlossaryElementSchema
from app.infrastructure.models.glossary_element import GlossaryElement


class GlossaryRepository(SQLAlchemyRepository, IGlossaryRepository):
    """Репозиторий для полной перезаписи таблицы glossary_element."""

    model = GlossaryElement
    response_dto = GlossaryElementSchema

    async def replace_all(
        self, elements: list[GlossaryElementCreateDTO], batch_size: int = 5000
    ) -> int:
        """Полностью заменяет данные глоссария в одной транзакции и обновляет materialized view."""
        lock_acquired = False
        try:
            lock_result = await self.session.execute(
                text("SELECT pg_try_advisory_lock(hashtext('glossary_sync'))")
            )
            lock_acquired = bool(lock_result.scalar())
            if not lock_acquired:
                raise ConflictError(
                    "Синхронизация глоссария уже выполняется в другом процессе"
                )

            await self.session.execute(
                text(
                    "ALTER TABLE glossary_element DISABLE TRIGGER trig_refresh_glossary"
                )
            )
            try:
                await self.session.execute(text("TRUNCATE TABLE glossary_element"))

                for idx in range(0, len(elements), batch_size):
                    batch = elements[idx : idx + batch_size]
                    if not batch:
                        continue

                    await self.session.execute(
                        insert(GlossaryElement).values(
                            [element.model_dump() for element in batch]
                        )
                    )

                await self.session.execute(text("REFRESH MATERIALIZED VIEW ge_splitted"))
            finally:
                await self.session.execute(
                    text(
                        "ALTER TABLE glossary_element ENABLE TRIGGER trig_refresh_glossary"
                    )
                )

            return len(elements)
        finally:
            if lock_acquired:
                await self.session.execute(
                    text("SELECT pg_advisory_unlock(hashtext('glossary_sync'))")
                )
