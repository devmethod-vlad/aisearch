import typing as tp

from celery.schedules import crontab

from app.settings.config import settings

SCHEDULED_TASKS = {
    "collect-all-data-from-confluence-daily": {
        "task": "collect-all-data-from-confluence",
        "schedule": crontab(
            hour=settings.app.knowledge_base_collect_data_time.hour,
            minute=settings.app.knowledge_base_collect_data_time.minute,
        ),
        "args": (),
    },
}


def get_schedule_config() -> dict[str, dict[str, tp.Any]]:
    """Получение конфигурации расписания"""
    return SCHEDULED_TASKS
