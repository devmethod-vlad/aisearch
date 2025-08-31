# Семантический поиск

<p>
    <a href="https://fastapi.tiangolo.com"><img alt="" src="https://img.shields.io/badge/FastAPI-009688.svg?style=flat&logo=FastAPI&logoColor=white"></a>
    <a href="https://milvus.io/"><img alt="" src="https://img.shields.io/badge/Milvus-00B7EB"></a>
    <a href="https://github.com/astral-sh/ruff"><img alt="" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
    <a href="https://github.com/reagento/dishka"><img alt="" src="https://img.shields.io/badge/DI-dishka-red"></a>
    <a href="https://pypi.org/project/pytest/"><img alt="" src="https://img.shields.io/badge/Pytest-green?logo=pytest"></a>
    <a href="https://github.com/psf/black"><img alt="" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

Для запуска проекта файл`.env` должен содержать следующие параметры:

    COMPOSE_PATH_SEPARATOR=';'
    COMPOSE_FILE='docker-compose.yml;docker-compose.dev.yml;docker-compose.gpu.yml'

    APP_MODE=dev
    APP_HOST=0.0.0.0
    APP_PORT=8000
    APP_DEBUG_HOST=0.0.0.0
    APP_DEBUG_PORT=5678
    APP_WORKERS_NUM=2
    APP_ACCESS_KEY=123
    APP_PREFIX=/some/prefix
    APP_LOGS_HOST_PATH=./logs/app
    APP_LOGS_CONTR_PATH=/usr/src/logs/app
    APP_MODEL_NAME="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    APP_KNOWLEDGE_BASE_MINITABLE_GOOGLE_LINK="https://docs.google.com/spreadsheets/d/13b7OQQW4galeI2tZc1XI43jzNch3gqwqP9HD0ofHoww/export?format=xlsx&gid=2126654700"
    APP_KNOWLEDGE_BASE_MEGATABLE_GOOGLE_LINK="https://docs.google.com/spreadsheets/d/1yYgNQG__nw7GYIvbPRqDxygsf747KeXgp1f9Q3sjaCg/export?format=xlsx&gid=0"
    APP_KNOWLEDGE_BASE_MAIN_HEADER_TEXT="База знаний"
    APP_KNOWLEDGE_BASE_TARGET_COLUMNS="Номер знания;Актуально;2 линия;Роль;Компонент;Описание ошибки;Анализ ошибки;Шаблон ответа;Для пользователя;Jira"
    APP_KNOWLEDGE_BASE_ROLES_SEPARATOR=";"
    APP_KNOWLEDGE_BASE_COLLECT_DATA_TIME=12:05 # hh:mm
    APP_CONFLUENCE_URL="https://wiki.mos.social"
    APP_CONFLUENCE_TOKEN=<token>
    APP_EDU_EMIAS_URL="https://edu.emias.ru"
    APP_EDU_EMIAS_TOKEN=<token>
    APP_EDU_EMIAS_ATTACHMENTS_PAGE_ID=223777111

    GUNICORN_LOGS_HOST_PATH=./logs/gunicorn
    GUNICORN_LOGS_CONTR_PATH=/usr/src/logs/gunicorn

    ETCD_AUTO_COMPACTION_MODE=revision # periodic
    ETCD_AUTO_COMPACTION_RETENTION=1000 # time, like "1h"
    ETCD_QUOTA_BACKEND_BYTES=4294967296
    ETCD_SNAPSHOT_COUNT=50000
    ETCD_HOST=aisearch-etcd
    ETCD_PORT=2379

    MINIO_ACCESS_KEY=minioadmin
    MINIO_SECRET_KEY=minioadmin
    MINIO_HOST=aisearch-minio
    MINIO_WEB_UI_PORT=9001
    MINIO_BUCKET_NAME=aisearch-bucket
    MINIO_PORT=9000

    MILVUS_HOST=aisearch-milvus
    MILVUS_PORT=19530
    MILVUS_WEB_UI_PORT=9091
    MILVUS_NLIST=128
    MILVUS_NPROBE=10
    MILVUS_LOAD_TIMEOUT=10 # секунды
    MILVUS_QUERY_TIMEOUT=2 # секунды

    REDIS_HOSTNAME=aisearch-redis
    REDIS_PORT=6379
    REDIS_DATABASE=4

    REDISINSIGHT_PORT=6399

    RESTRICT_MAX_CACHE_TTL=    # секунды
    RESTRICT_QUEUE_KEY=celery:pending-tasks-ids
    RESTRICT_BASE_CACHE_KEY=celery:cache-result

    RESTRICT_SEMANTIC_SEARCH_LAST_QUERY_TIME_KEY=celery:semantic-search:last-query-time
    RESTRICT_SEMANTIC_SEARCH_TIMEOUT_INTERVAL=2 # секунды
    RESTRICT_SEMANTIC_SEARCH_QUEUE_SIZE=3

    CELERY_LOGS_HOST_PATH=./logs/celery
    CELERY_LOGS_CONTR_PATH=/usr/src/logs/celery
    CELERY_WORKERS_NUM=2

##  Pre-commit и Ruff

Для использования Ruff:

    ruff check --fix

При выполнении команды с флагом --fix автоматически исправит мелкие недочеты вроде отступов и пробелов, порядка импортов

Для использования pre-commit:

Устанавливаем на компьютер через CMD

    pip install pre-commit

Внутри проекта в терминале IDE в установленном виртуальном окружении

    pre-commit install

Ожидаемый результат: pre-commit installed at .git\hooks\pre-commit

На случай обновлений периодически выполнять

    pre-commit autoupdate

Перед коммитом в терминале выполнить команду

    pre-commit run -a

Необходимо выполнять ее после правок до удачного прохождения всех стадий.
В случае невозможности на данном этапе привести все в порядок - закомментировать непроходимый этап в .pre-commit-config.yaml или удалить pre-commit

    pre-commit uninstall

Если правило форматирования в конкретном случае выполнить невозможно - сбоку от "проблемной строки" оставляем коммент вида '# noqa: <Код правила>'
