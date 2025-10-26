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
    COMPOSE_FILE='docker-compose.yml;docker-compose.dev.yml'

    CUDA_IMAGE=nvidia/cuda:12.8.0-devel-ubuntu22.04
    CUDA_WHEEL=cu128
    HTTPS_PROXY=''
    HTTP_PROXY=''
    IMAGE_NAME_BASE=cuda-python-uv-base:latest
    IMAGE_NAME_DEPS=aisearch-deps-base:latest
    IMAGE_NAME=aisearch-app-dev:latest
    REGISTRY=localhost:5001

    APP_MODE=dev
    APP_HOST=0.0.0.0
    APP_PORT=5155
    APP_DEBUG_HOST=0.0.0.0
    APP_DEBUG_PORT=5678
    APP_WORKERS_NUM=2
    APP_ACCESS_KEY=123
    APP_PREFIX=
    APP_LOGS_HOST_PATH=./logs/app
    APP_LOGS_CONTR_PATH=/usr/src/logs/app
    APP_USE_CACHE=false  # Использовать ли кэширование
    APP_MODELSTORE_HOST_PATH=C:/Users/omka/models
    APP_MODELSTORE_CONTR_PATH=/usr/src/models
    APP_NORMALIZE_QUERY=true # Нормализовать ли запрос

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
    MILVUS_USE_SSL=false
    MILVUS_WEB_UI_PORT=9091
    MILVUS_CONNECTION_TIMEOUT=120
    MILVUS_QUERY_TIMEOUT=30
    MILVUS_METRIC_TYPE=IP
    MILVUS_COLLECTION_NAME=kb_default
    MILVUS_RECREATE_COLLECTION=false
    MILVUS_VECTOR_FIELD=embedding
    MILVUS_ID_FIELD=pk
    MILVUS_SEARCH_FIELDS=question
    MILVUS_OUTPUT_FIELDS=ext_id,question,analysis,answer
    MILVUS_VOLUME_HOST_PATH=./volumes/milvus
    MILVUS_VOLUME_CONTR_PATH=/var/lib/milvus
    MILVUS_MODEL=USER-bge-m3
    MILVUS_SCHEMA_PATH=app/config/conf.json

    # === Очередь LLM (для отложенной генерации, если нет слота семафора) ===
    LLM_QUEUE_LIST_KEY=llm:queue:list
    LLM_QUEUE_TICKET_HASH_PREFIX=llm:ticket:
    LLM_QUEUE_MAX_SIZE=10
    LLM_QUEUE_TICKET_TTL=3600
    LLM_QUEUE_DRAIN_INTERVAL_SEC=1

    # === Переключатели поиска ===
    SEARCH_USE_HYBRID=true        # если false -> только dense + (опц.) reranker
    SEARCH_USE_OPENSEARCH=true    # взаимоисключимо с BM25 (приоритет OS)
    SEARCH_USE_BM25=false
    SEARCH_USE_RERANKER=true

    # === Параметры гибридного склейщика ===
    HYBRID_DENSE_TOP_K=20
    HYBRID_LEX_TOP_K=50
    HYBRID_TOP_K=5
    HYBRID_W_CE=0.6
    HYBRID_W_DENSE=0.25
    HYBRID_W_LEX=0.15
    HYBRID_DENSE_THRESHOLD=0.0
    HYBRID_LEX_THRESHOLD=0.0
    HYBRID_CE_THRESHOLD=0.0
    HYBRID_CACHE_TTL=3600
    HYBRID_VERSION=v1
    HYBRID_COLLECTION_NAME=kb_default
    HYBRID_MERGE_BY_FIELD=ext_id
    HYBRID_OUTPUT_FIELDS=ext_id,question,analysis,answer
    HYBRID_DENSE_ABS_MIN=0.25
    HYBRID_DENSE_REL_MIN=0.6
    HYBRID_LEX_REL_MIN=0.5
    HYBRID_PRECUT_MIN_KEEP=3

    # === SlowAPI ===
    
    SLOWAPI_LIMIT_SEARCH=5/minute
    SLOWAPI_LIMIT_GENERATE=5/minute


    # === OpenSearch ===
    OS_HOST=aisearch-opensearch
    OS_PORT=9200
    OS_INDEX_NAME=aisearch-qa
    OS_USE_SSL=false
    OS_VERIFY_CERTS=false
    OS_USER=
    OS_PASSWORD=
    OS_SEARCH_FIELDS=question,analysis,answer
    OS_OUTPUT_FIELDS=ext_id,question,analysis,answer
    OS_OPERATOR=or
    OS_MIN_SHOULD_MATCH=1
    OS_FUZZINESS=0
    OS_USE_RESCORE=false
    OS_INDEX_ANSWER=true
    OS_BULK_CHUNK_SIZE=1000
    OS_RECREATE_INDEX=true
    OS_SCHEMA_PATH=app/config/os_index.json
    OS_VOLUME_HOST_PATH=./volumes/opensearch-data
    OS_VOLUME_CONTR_PATH=/usr/share/opensearch/data
    
    NLTK_DATA_HOST_PATH=E:/nltk    # выкачка ресурсов через python -m nltk.downloader -d путь_к_папке punkt stopwords punkt_tab
    NLTK_DATA_CONTR_PATH=/srv/nltk_data
    

    # === BM25 (Whoosh) ===
    BM25_INDEX_PATH_HOST=C:/Users/omka/models/1
    BM25_INDEX_PATH=/usr/src/bm25index
    BM25_SCHEMA_FIELDS=ext_id,question,analysis,answer
    BM25_OUTPUT_FIELDS=ext_id,question,analysis,answer
    BM25_RECREATE_INDEX=false

    REDIS_HOSTNAME=redis
    REDIS_PORT=6379
    REDIS_DATABASE=7

    REDISINSIGHT_PORT=6399

    CELERY_LOGS_HOST_PATH=./logs/celery
    CELERY_LOGS_CONTR_PATH=/usr/src/logs/celery
    CELERY_LOGS_QUEUE_HOST_PATH=./logs/queue
    CELERY_LOGS_QUEUE_CONTR_PATH=/usr/src/logs/queue
    CELERY_WORKERS_NUM=2

    # === Глобальный семафор (общий лимит конкаренси для поиска и генерации) ===
    LLM_GLOBAL_SEM_REDIS_DSN="redis://${REDIS_HOSTNAME}:${REDIS_PORT}/${REDIS_DATABASE}"
    LLM_GLOBAL_SEM_KEY=llm:{global}:sem
    LLM_GLOBAL_SEM_LIMIT=2
    LLM_GLOBAL_SEM_TTL_MS=120000
    LLM_GLOBAL_SEM_WAIT_TIMEOUT_MS=30000
    LLM_GLOBAL_SEM_HEARTBEAT_MS=30000

    # === CrossEncoder (reranker) ===
    RERANKER_DEVICE=cuda   # или cuda
    RERANKER_BATCH_SIZE=128
    RERANKER_MAX_LENGTH=192
    RERANKER_DTYPE=fp16
    RERANKER_SCORE_MOD=sigmoid
    RERANKER_SOFTMAX_TEMP=0.7
    RERANKER_MODEL=cross-encoder-russian-msmarco
    RERANKER_PAIRS_FIELDS=question,analysis,answer

    # === vLLM (локальный OpenAI-совместимый сервер) ===
    VLLM_BASE_URL=http://aisearch-vllm:8000/v1
    VLLM_PORT=8003
    VLLM_API_KEY=local-dev
    VLLM_MODEL_SERVED_NAME=mistral-nemo-awq
    VLLM_MAX_INPUT_TOKENS=4096
    VLLM_MAX_OUTPUT_TOKENS=512
    VLLM_TEMPERATURE=0.7
    VLLM_TOP_P=0.9
    VLLM_REQUEST_TIMEOUT=60
    VLLM_STREAM=false
    VLLM_MODEL_NAME=Llama-3.2-1B-Instruct-NEO-SI-FI-GGUF/Llama-3.2-1B-Instruct-NEO-SI-FI-Q4_K_M-imat.gguf
    VLLM_GPU_MEMORY_UTILIZATION=0.6

    # === Прогрев эмбеддера/реранкера при старте ===
    WARMUP_ENABLED=true
    WARMUP_CE_PAIRS="Как оформить доступ в систему?||Как оформить доступ врача в систему?;Запись к врачу||Как записаться к врачу"
    WARMUP_EMBED_TEXTS="Пример вопроса один;Пример вопроса два"

    TIMEIT_LOG_METRICS_ENABLED=true  # включить логгирование времени
    TIMEIT_RESPONSE_METRIC_ENABLED=true # включить метрики времени в результат задачи

## Сборка образов
Для ускорения процесса работы разделим сборку на три образа:
- первый - это nvidia-cuda+python+uv. Собирается, фактически, один раз.
- второй образ строим на основе первого, в него ставим зависимости.
- третий образ - рабочий, на основе второго. При его сборке поверх второго, uv делает проверку и устанавливает/удаляет только те пакеты
которых еще нет, либо уже пропали. Таким образом, при изменении легких зависимостей процесс сборки должен происходить быстро. При изменении/добавлении тяжелых зависимостей нужно второй образ.

## Сборка на примере базового образа для Bash
```
( set -a; . <(sed 's/\r$//' .env); set +a;
  DOCKER_BUILDKIT=0 docker build -f Dockerfile_search_base -t "$IMAGE_NAME_BASE" \
    --build-arg http_proxy="$HTTP_PROXY" --build-arg https_proxy="$HTTPS_PROXY" \
    --build-arg HTTP_PROXY="$HTTP_PROXY" --build-arg HTTPS_PROXY="$HTTPS_PROXY" . && \
  docker tag "$IMAGE_NAME_BASE" "$REGISTRY/$IMAGE_NAME_BASE" && \
  docker push "$REGISTRY/$IMAGE_NAME_BASE"
)
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

  docker build -f Dockerfile_search_base -t $env:IMAGE_NAME_BASE `
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
