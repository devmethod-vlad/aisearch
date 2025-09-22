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

    HTTPS_PROXY=''
    HTTP_PROXY=''
    IMAGE_NAME_BASE=cuda-python-uv-base:latest
    IMAGE_NAME_DEPS=aisearch-deps-base:latest
    IMAGE_NAME=aisearch-app-dev:latest
    REGISTRY=localhost:5001

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
    APP_MODELSTORE_HOST_PATH=./models
    APP_MODELSTORE_CONTR_PATH=/usr/src/models

    BM25_INDEX_PATH_HOST=./bm25index
    BM25_INDEX_PATH=/usr/src/bm25index

    GUNICORN_LOGS_HOST_PATH=./logs/gunicorn
    GUNICORN_LOGS_CONTR_PATH=/usr/src/logs/gunicorn

    ETCD_AUTO_COMPACTION_MODE=revision # periodic
    ETCD_AUTO_COMPACTION_RETENTION=1000 # time, like "1h"
    ETCD_QUOTA_BACKEND_BYTES=4294967296
    ETCD_SNAPSHOT_COUNT=50000
    ETCD_HOST=aisearch-etcd
    ETCD_PORT=2379
    ETCD_VOLUME_HOST_PATH=./volumes/etcd
    ETCD_VOLUME_CONTR_PATH=/etcd

    MINIO_ACCESS_KEY=minioadmin
    MINIO_SECRET_KEY=minioadmin
    MINIO_HOST=aisearch-minio
    MINIO_WEB_UI_PORT=9001
    MINIO_BUCKET_NAME=aisearch-bucket
    MINIO_PORT=9000
    MINIO_VOLUME_HOST_PATH=./volumes/minio
    MINIO_VOLUME_CONTR_PATH=/minio_data

    MILVUS_HOST=aisearch-milvus
    MILVUS_PORT=19530
    MILVUS_WEB_UI_PORT=9091
    MILVUS_NLIST=128
    MILVUS_NPROBE=10
    MILVUS_LOAD_TIMEOUT=10 # секунды
    MILVUS_QUERY_TIMEOUT=2 # секунды
    MILVUS_VOLUME_HOST_PATH=./volumes/milvus
    MILVUS_VOLUME_CONTR_PATH=/var/lib/milvus

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

    VLLM_MODEL_SERVED_NAME=mistral-nemo-awq

## Сборка образов
Для ускорения процесса работы разделим сборку на три образа:
- первый - это nvidia-cuda+python+uv. Собирается, фактически, один раз.
- второй образ строим на основе первого, в него ставим зависимости.
- третий образ - рабочий, на основе второго. При его сборке поверх второго, uv делает проверку и устанавливает/удаляет только те пакеты
которых еще нет, либо уже пропали. Таким образом, при изменении легких зависимостей процесс сборки должен происходить быстро. При изменении/добавлении тяжелых зависимостей нужно второй образ.

## Сборка на примере базового образа для Bash
```
( set -a; . ./.env; set +a; docker build -f Dockerfile_base -t "$IMAGE_NAME_BASE" --build-arg http_proxy="$HTTP_PROXY" --build-arg https_proxy="$HTTPS_PROXY" --build-arg HTTP_PROXY="$HTTP_PROXY" --build-arg HTTPS_PROXY="$HTTPS_PROXY" . && docker tag "$IMAGE_NAME_BASE" "$REGISTRY/$IMAGE_NAME_BASE" && docker push "$REGISTRY/$IMAGE_NAME_BASE" )
```

## Сборка на примере базового образа для PowerShell
```
& {
  Get-Content .env |
    Where-Object { $_ -match '^\s*[^#]+=' } |
    ForEach-Object {
      $n,$v = $_ -split '=', 2
      Set-Item Env:$n ($v.Trim([char]34,[char]39,[char]96).Trim())
    }

  docker build -f Dockerfile_base -t $env:IMAGE_NAME_BASE `
    --build-arg http_proxy=$env:HTTP_PROXY `
    --build-arg https_proxy=$env:HTTPS_PROXY `
    --build-arg HTTP_PROXY=$env:HTTP_PROXY `
    --build-arg HTTPS_PROXY=$env:HTTPS_PROXY `
    .;
  docker tag  $env:IMAGE_NAME_BASE  $env:REGISTRY/$env:IMAGE_NAME_BASE;
  docker push $env:REGISTRY/$env:IMAGE_NAME_BASE
}
```

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
