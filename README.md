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

    REQUIRED_CONTAINERS="aisearch-celery-search-worker aisearch-queue-worker"

    LOGS_HOST_DIR=./logs
    LOGS_CONTR_DIR=/usr/src/logs
    LOGS_ACCESS_LOCATION=/app/access.log

    CUDA_IMAGE=nvidia/cuda:12.8.0-devel-ubuntu22.04
    CUDA_WHEEL=cu128
    HTTPS_PROXY_ENV=''
    HTTP_PROXY_ENV=''
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
    APP_LOG_LEVEL=info
    APP_PREFIX=
    APP_LOGS_LOCATION=app/app.log
    APP_LOG_LEVEL=INFO
    APP_MODELSTORE_HOST_PATH=C:/Users/omka/models
    APP_MODELSTORE_CONTR_PATH=/usr/src/models
    APP_NORMALIZE_QUERY=true # Нормализовать ли запрос
    APP_COLLECTION_FILES_HOST_DIR=./volumes/collections
    APP_COLLECTION_FILES_CONTR_DIR=/collections
    APP_DATA_UNIQUE_ID=ext_id
    APP_RECREATE_DATA=true
    APP_GENERATE_PRELAUNCH_DATA=true
    APP_PRELAUNCH_DATA_HOST_DIR=./volumes/prelaunch
    APP_PRELAUNCH_DATA_CONTR_DIR=/prelaunch

    # === CORS ===
    APP_CORS_ENABLED=false
    APP_CORS_ALLOW_ORIGINS=
    APP_CORS_ALLOW_ORIGIN_REGEX=
    APP_CORS_ALLOW_METHODS=GET,POST,PUT,PATCH,DELETE,OPTIONS
    APP_CORS_ALLOW_HEADERS=*
    APP_CORS_ALLOW_CREDENTIALS=false

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
    MILVUS_VECTOR_FIELD=embedding
    MILVUS_ID_FIELD=pk
    MILVUS_SEARCH_FIELDS=question
    MILVUS_OUTPUT_FIELDS=row_idx,source,ext_id,page_id,role,component,question,analysis,answer,answer_copy,jira
    MILVUS_VOLUME_HOST_PATH=./volumes/milvus
    MILVUS_VOLUME_CONTR_PATH=/var/lib/milvus
    MILVUS_MODEL_NAME=USER-bge-m3
    MILVUS_SCHEMA_PATH=app/settings/conf.json

    # === Очередь фоновых поисковых задач (legacy-имя LLM_QUEUE_* для совместимости) ===
    LLM_QUEUE_LIST_KEY=llm:queue:list
    LLM_QUEUE_TICKET_HASH_PREFIX=llm:ticket:
    LLM_QUEUE_MAX_SIZE=10
    LLM_QUEUE_TICKET_TTL=3600
    LLM_QUEUE_DRAIN_INTERVAL_SEC=1

    # === Переключатели поиска ===
    SEARCH_USE_OPENSEARCH=true

    # === Финальный ранжир гибридного поиска ===
    HYBRID_FINAL_RANK_MODE=fusion_only
    HYBRID_FINAL_W_FUSION=1.0
    HYBRID_FINAL_W_CE=0.0
    HYBRID_FINAL_FUSION_NORM=max
    HYBRID_FINAL_CE_SCORE=processed

    # === Параметры гибридного склейщика ===
    HYBRID_DENSE_TOP_K=20
    HYBRID_LEX_TOP_K=50
    HYBRID_TOP_K=5
    HYBRID_W_DENSE=0.25
    HYBRID_W_LEX=0.15
    HYBRID_FUSION_MODE=weighted_score
    HYBRID_RRF_K=60
    HYBRID_CACHE_TTL=3600
    HYBRID_VERSION=v1
    HYBRID_COLLECTION_NAME=kb_default
    HYBRID_MERGE_BY_FIELD=ext_id
    HYBRID_MERGE_FIELDS=row_idx,source,ext_id,page_id,role,component,question,analysis,answer,answer_copy,jira
    HYBRID_OUTPUT_FIELDS=row_idx,source,ext_id,page_id,role,component,question,analysis,answer,answer_copy,jira
    HYBRID_DENSE_ABS_MIN=0.25
    HYBRID_DENSE_REL_MIN=0.6
    HYBRID_LEX_REL_MIN=0.5
    HYBRID_PRECUT_MIN_KEEP=3
    HYBRID_INTERMEDIATE_RESULTS_TOP_K=5

    # === SlowAPI ===

    SLOWAPI_LIMIT_SEARCH=5/minute


    # === OpenSearch ===
    OS_HOST=aisearch-opensearch
    OS_PORT=9200
    OS_INDEX_NAME=aisearch-qa
    OS_USE_SSL=false
    OS_VERIFY_CERTS=false
    OS_USER=
    OS_PASSWORD=
    OS_SEARCH_FIELDS=question,analysis,answer
    OS_OUTPUT_FIELDS=row_idx,source,ext_id,page_id,role,component,question,analysis,answer,answer_copy,jira
    OS_OPERATOR=or
    # minimum_should_match применяется только при OS_OPERATOR=or
    OS_MIN_SHOULD_MATCH=1
    # best_fields | most_fields | cross_fields | phrase | phrase_prefix | bool_prefix
    OS_MULTI_MATCH_TYPE=best_fields
    OS_FUZZINESS=0
    OS_USE_RESCORE=false
    OS_INDEX_ANSWER=true
    OS_BULK_CHUNK_SIZE=1000
    OS_SCHEMA_PATH=app/settings/os_index.json
    OS_VOLUME_HOST_PATH=./volumes/opensearch-data
    OS_VOLUME_CONTR_PATH=/usr/share/opensearch/data

    NLTK_DATA_HOST_PATH=E:/nltk    # выкачка ресурсов через python -m nltk.downloader -d путь_к_папке punkt stopwords punkt_tab
    NLTK_DATA_CONTR_PATH=/srv/nltk_data

    REDIS_HOSTNAME=redis
    REDIS_PORT=6379
    REDIS_DATABASE=7

    REDISINSIGHT_PORT=6399

    CELERY_LOGS_LOCATION=/celery/celery.log
    CELERY_LOG_LEVEL=INFO
    CELERY_LOGS_QUEUE_LOCATION=/queue/queue.log
    CELERY_LOG_QUEUE_LEVEL=INFO
    CELERY_WORKERS_NUM=2
    CELERY_GRACE_PERIOD_SECONDS=15

    # === Глобальный семафор поисковых задач (legacy-имя LLM_GLOBAL_SEM_* для совместимости) ===
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
    RERANKER_MODEL_NAME=cross-encoder-russian-msmarco
    RERANKER_PAIRS_FIELDS=question,analysis,answer
    # === Прогрев эмбеддера/реранкера при старте ===
    WARMUP_ENABLED=true
    WARMUP_CE_PAIRS="Как оформить доступ в систему?||Как оформить доступ врача в систему?;Запись к врачу||Как записаться к врачу"
    WARMUP_EMBED_TEXTS="Пример вопроса один;Пример вопроса два"

    TIMEIT_LOG_METRICS_ENABLED=true  # включить логгирование времени
    # Метрики в response включаются runtime-параметром metrics_enable в /hybrid-search/search

    EXTRACT_EDU_EMIAS_URL="https://edu.emias.ru"
    EXTRACT_EDU_EMIAS_TOKEN=""
    EXTRACT_EDU_EMIAS_ATTACHMENTS_PAGE_ID=223792475
    EXTRACT_EDU_TIMEOUT=20
    EXTRACT_KNOWLEDGE_BASE_FILE_NAME="KB_wiki.xlsx"
    EXTRACT_VIO_BASE_FILE_NAME="Вопросы_и_ответы.xlsx"
    EXTRACT_CRON_UPDATE_TIMES="3:00, 7:00"
    EXTRACT_LOGS_LOCATION=/updater/updater.log
    EXTRACT_LOG_LEVEL=INFO
    EXTRACT_BASE_HARVESTER_API_URL="https://edu.emias.ru/edu-rest-api/test/data-harvester"
    EXTRACT_VIO_HARVESTER_SUFFIX="/vio/runtime_harvest"
    EXTRACT_KB_HARVESTER_SUFFIX="/knowledge-base/collect-all"

    # === Glossary sync ===
    GLOSSARY_API_URL="https://edu.emias.ru/edu-rest-api/test/glossary/glossary/getelements"
    GLOSSARY_PAGE_LIMIT=500
    GLOSSARY_CRON_UPDATE_TIMES="03:30"
    GLOSSARY_REQUEST_TIMEOUT=30
    GLOSSARY_MAX_RETRIES=5
    GLOSSARY_RETRY_BACKOFF_BASE_SECONDS=1
    GLOSSARY_RETRY_BACKOFF_MAX_SECONDS=30
    GLOSSARY_ABBREVIATION_DELIMITER=";"
    GLOSSARY_TERM_DELIMITER=";"

    # === PosgreSQL ===
    POSTGRES_USER=postgres
    POSTGRES_PASSWORD=111
    POSTGRES_HOST=aisearch-db
    POSTGRES_PORT=5432
    POSTGRES_DB=aisearch
    POSTGRES_VOLUME_HOST_PATH=./volumes/pgdata
    POSTGRES_VOLUME_CONTR_PATH=/var/lib/postgresql/data

    PGADMIN_DEFAULT_EMAIL=pgadmin@example.com
    PGADMIN_DEFAULT_PASSWORD=admin
    PGADMIN_PORT=5050

    SHORT_DENSE_TOP_K=20
    SHORT_LEX_TOP_K=50
    SHORT_TOP_K=5
    SHORT_W_DENSE=0.25
    SHORT_W_LEX=0.15
    SHORT_FUSION_MODE=weighted_score
    SHORT_RRF_K=60
    SHORT_USE_OPENSEARCH=true
    SHORT_MODE=true
    SHORT_MODE_LIMIT=4

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

## Установка всех зависимостей в единое окружение (Windows, cmd)
```
uv pip compile pyproject.toml --extra api --extra search --extra queue --extra dev -o requirements.txt && uv pip sync requirements.txt --system && del requirements.txt
```
или
```
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 && uv pip compile pyproject.toml --extra api --extra search --extra queue --extra dev -o requirements.txt && uv pip sync requirements.txt && del requirements.txt
```
## Сборка Dockerfile_search_base + Dockerfile_search_deps + Dockerfile_search_main (Windows, cmd)
Поднимать локальный registry c
```
REGISTRY_STORAGE_DELETE_ENABLED=true
```

Базовый образ (Dockerfile_search_base)
```
powershell -Command "& { Get-Content .env | Where-Object { $_ -match '^\s*[^#]+=' } | ForEach-Object { $n,$v = $_ -split '=', 2; Set-Item Env:$n ($v.Trim([char]34,[char]39,[char]96).Trim()) }; docker build -f Dockerfile_search_base -t $env:IMAGE_NAME_BASE --build-arg http_proxy=$env:HTTP_PROXY --build-arg https_proxy=$env:HTTPS_PROXY --build-arg HTTP_PROXY=$env:HTTP_PROXY --build-arg HTTPS_PROXY=$env:HTTPS_PROXY .; docker tag $env:IMAGE_NAME_BASE $env:REGISTRY/$env:IMAGE_NAME_BASE; $old_digest = docker inspect --format='{{index .RepoDigests 0}}' ($env:REGISTRY + '/' + $env:IMAGE_NAME_BASE) 2>$null; if ($old_digest) { $old_digest = $old_digest.split('@')[1]; try { Invoke-RestMethod -Uri ('http://' + $env:REGISTRY + '/v2/' + $env:IMAGE_NAME_BASE + '/manifests/' + $old_digest) -Method Delete } catch { if ($_.Exception.Response.StatusCode -ne 'NotFound') { throw } } }; docker push $env:REGISTRY/$env:IMAGE_NAME_BASE; docker rmi $env:IMAGE_NAME_BASE -f 2>$null; }"
```

Крупные зависимости (Dockerfile_search_deps)
```
powershell -Command "& { Get-Content .env | Where-Object { $_ -match '^\s*[^#]+=' } | ForEach-Object { $n,$v = $_ -split '=', 2; Set-Item Env:$n ($v.Trim([char]34,[char]39,[char]96).Trim()) }; docker build -f Dockerfile_search_deps -t $env:IMAGE_NAME_DEPS --build-arg http_proxy=$env:HTTP_PROXY --build-arg https_proxy=$env:HTTPS_PROXY --build-arg HTTP_PROXY=$env:HTTP_PROXY --build-arg HTTPS_PROXY=$env:HTTPS_PROXY --build-arg CUDA_WHEEL=$env:CUDA_WHEEL .; docker tag $env:IMAGE_NAME_DEPS $env:REGISTRY/$env:IMAGE_NAME_DEPS; $old_digest = docker inspect --format='{{index .RepoDigests 0}}' ($env:REGISTRY + '/' + $env:IMAGE_NAME_DEPS) 2>$null; if ($old_digest) { $old_digest = $old_digest.split('@')[1]; try { Invoke-RestMethod -Uri ('http://' + $env:REGISTRY + '/v2/' + $env:IMAGE_NAME_DEPS + '/manifests/' + $old_digest) -Method Delete } catch { if ($_.Exception.Response.StatusCode -ne 'NotFound') { throw } } }; docker push $env:REGISTRY/$env:IMAGE_NAME_DEPS; docker rmi $env:IMAGE_NAME_DEPS -f 2>$null; }"
```

Остальные зависимости (Dockerfile_search_main, поменять на Dockerfile_search_test по желанию)
```
powershell -Command "& { Get-Content .env | Where-Object { $_ -match '^\s*[^#]+=' } | ForEach-Object { $n,$v = $_ -split '=', 2; Set-Item Env:$n ($v.Trim([char]34,[char]39,[char]96).Trim()) }; docker build -f Dockerfile_search_main -t $env:IMAGE_NAME --build-arg http_proxy=$env:HTTP_PROXY --build-arg https_proxy=$env:HTTPS_PROXY --build-arg HTTP_PROXY=$env:HTTP_PROXY --build-arg HTTPS_PROXY=$env:HTTPS_PROXY .; docker tag $env:IMAGE_NAME $env:REGISTRY/$env:IMAGE_NAME; $old_digest = docker inspect --format='{{index .RepoDigests 0}}' ($env:REGISTRY + '/' + $env:IMAGE_NAME) 2>$null; if ($old_digest) { $old_digest = $old_digest.split('@')[1]; try { Invoke-RestMethod -Uri ('http://' + $env:REGISTRY + '/v2/' + $env:IMAGE_NAME + '/manifests/' + $old_digest) -Method Delete } catch { if ($_.Exception.Response.StatusCode -ne 'NotFound') { throw } } }; docker push $env:REGISTRY/$env:IMAGE_NAME; docker rmi $env:IMAGE_NAME -f 2>$null; }"
```
## Установка CUDA
Установить [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive).

Добавить путь CUDA в PATH
```
echo 'export PATH=/usr/local/cuda/bin:${PATH}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}' >> ~/.bashrc
bashrc && source ~/.bashrc
```

Проверить
```
nvcc --version
```
## Запуск pre_launch только для экспорта данных
Запустить pre_launch.py в контейнере aisearch-celery-worker
```
python3 pre_launch.py --export-only
```


### Runtime-параметры поиска в API
Параметры `search_use_cache`, `show_intermediate_results`, `metrics_enable` и `presearch.field` теперь управляются только телом запроса `/hybrid-search/search`, а не env-переменными.
`metrics_enable` по умолчанию `false`: при `false` блок `metrics` в payload результата отсутствует, при `true` — добавляется после завершения задачи.
Параметры `HYBRID_W_DENSE` и `HYBRID_W_LEX` используются только на стадии retrieval fusion: в `weighted_score` — как веса score, в `rrf` не используются; для RRF применяются `HYBRID_RRF_W_DENSE` и `HYBRID_RRF_W_LEX`.
Когда `SHORT_MODE=true` и нормализованный запрос содержит не больше `SHORT_MODE_LIMIT` токенов, short-mode переопределяет обычные настройки:
- `SHORT_DENSE_TOP_K` → `HYBRID_DENSE_TOP_K`
- `SHORT_LEX_TOP_K` → `HYBRID_LEX_TOP_K`
- `SHORT_TOP_K` → `HYBRID_TOP_K`, если `top_k` не передан в request body
- `SHORT_W_DENSE` → `HYBRID_W_DENSE`
- `SHORT_W_LEX` → `HYBRID_W_LEX`
- `SHORT_RRF_W_DENSE` → `HYBRID_RRF_W_DENSE`
- `SHORT_RRF_W_LEX` → `HYBRID_RRF_W_LEX`
- `SHORT_FUSION_MODE` → `HYBRID_FUSION_MODE`
- `SHORT_RRF_K` → `HYBRID_RRF_K`
- `SHORT_USE_OPENSEARCH` → `SEARCH_USE_OPENSEARCH`
- `SHORT_FINAL_W_FUSION` → `HYBRID_FINAL_W_FUSION`
- `SHORT_FINAL_W_CE` → `HYBRID_FINAL_W_CE`
- `SHORT_FINAL_FUSION_NORM` → `HYBRID_FINAL_FUSION_NORM`
- `SHORT_FINAL_CE_SCORE` → `HYBRID_FINAL_CE_SCORE`

Нюансы при short-mode:
- `top_k` из request body имеет приоритет над `SHORT_TOP_K` и `HYBRID_TOP_K`;
- `SHORT_RRF_K` влияет только при `SHORT_FUSION_MODE=rrf`;
- если `SEARCH_USE_OPENSEARCH=false` или `SHORT_USE_OPENSEARCH=false`, lexical-ветка отключена и RRF работает только по доступным candidate lists.

Финальное ранжирование управляется `HYBRID_FINAL_RANK_MODE` (`fusion_only|ce_final|ce_blend|legacy_weighted`). В short-mode для финального этапа используются переопределения `SHORT_FINAL_W_FUSION`, `SHORT_FINAL_W_CE`, `SHORT_FINAL_FUSION_NORM`, `SHORT_FINAL_CE_SCORE`.


## Локальная проверка CORS для frontend (Vite)

Если frontend запущен на `http://localhost:5173`, включите CORS в `.env`:

```env
APP_CORS_ENABLED=true
APP_CORS_ALLOW_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
APP_CORS_ALLOW_METHODS=GET,POST,OPTIONS
APP_CORS_ALLOW_HEADERS=*
APP_CORS_ALLOW_CREDENTIALS=false
```

После изменения `.env` перезапустите backend-контейнер:

```bash
docker compose restart aisearch-app
```

или, при необходимости пересборки:

```bash
docker compose up -d --build aisearch-app
```

Ручная проверка preflight-запроса:

```bash
curl -i -X OPTIONS "http://127.0.0.1:5155/hybrid-search/search" \
  -H "Origin: http://localhost:5173" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: content-type"
```

В ответе должны появиться CORS-заголовки, например:
`access-control-allow-origin: http://localhost:5173`.
