from datetime import UTC, datetime, timedelta

from dateutil.relativedelta import relativedelta
from pydantic import Field, computed_field

from app.common.arbitrary_model import ArbitraryModel


class TimeIntervalRequest(ArbitraryModel):
    """Параметры временного интервала"""

    interval: str | None = Field(
        default=None,
        description="Временной интервал (1h, 2d, 3w, 1m, 1y)",
        pattern=r"^([1-9]\d{0,2})([hdwmy])$",
    )

    @computed_field
    @property
    def start_time(self) -> datetime | None:
        """Вычисляемое поле start_time на основе interval"""
        if not self.interval:
            return None

        value = int(self.interval[:-1])
        unit = self.interval[-1].lower()
        now = datetime.now(tz=UTC)

        if unit == "h":
            return now - timedelta(hours=value)
        elif unit == "d":
            return now - timedelta(days=value)
        elif unit == "w":
            return now - timedelta(weeks=value)
        elif unit == "m":
            return now - relativedelta(months=value)
        elif unit == "y":
            return now - relativedelta(years=value)

        return None


class PaginatedRequest(ArbitraryModel):
    """Пагинированный запрос"""

    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
