from sqlalchemy import func
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, mapped_column


class Base(DeclarativeBase):
    created_at = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    modified_at = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        server_onupdate=func.now(),
        onupdate=func.now(),
    )
