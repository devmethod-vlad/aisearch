import uuid

from sqlalchemy import String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.schema import Index, UniqueConstraint

from app.infrastructure.models.base import Base


class Role(Base):
    __tablename__ = "role"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(500), nullable=False, default="")

    __table_args__ = (
        UniqueConstraint("name", name="unique_role_name"),
        Index(
            "idx_role_name",
            "name",
            postgresql_using="btree",
        ),
        Index(
            "idx_role_name_gin_trgm",
            "name",
            postgresql_using="gin",
            postgresql_ops={"name": "gin_trgm_ops"},
        ),
        Index(
            "idx_role_created_at_asc",
            "created_at",
            postgresql_using="btree",
            postgresql_ops={"created_at": "ASC"},
        ),
        Index(
            "idx_role_created_at_desc",
            "created_at",
            postgresql_using="btree",
            postgresql_ops={"created_at": "DESC"},
        ),
        Index(
            "idx_role_modified_at_asc",
            "modified_at",
            postgresql_using="btree",
            postgresql_ops={"modified_at": "ASC"},
        ),
        Index(
            "idx_role_modified_at_desc",
            "modified_at",
            postgresql_using="btree",
            postgresql_ops={"modified_at": "DESC"},
        ),
    )
