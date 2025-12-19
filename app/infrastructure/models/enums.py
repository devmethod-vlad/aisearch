from sqlalchemy import Enum

from app.domain.schemas.enums import RatingEnum

RatingEnumType = Enum(RatingEnum, name="rating_enum")
