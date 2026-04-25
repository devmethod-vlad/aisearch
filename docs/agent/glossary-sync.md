# Синхронизация глоссария аббревиатур из внешнего API

## 1. Зачем сделаны изменения

В проект добавлена инфраструктура для хранения и регулярной синхронизации глоссария аббревиатур.

Глоссарий нужен как отдельный справочник, который в следующих задачах можно будет использовать для enrichment:

- пользовательских запросов перед поиском;
- материалов базы знаний перед индексированием;
- дополнительных поисковых сценариев по аббревиатурам, терминам и определениям.

До этого PR в проекте не было отдельного слоя для хранения аббревиатур. Данные глоссария должны приходить из внешнего API, но использоваться внутри проекта как локальная PostgreSQL-таблица, чтобы runtime-логика поиска не зависела напрямую от доступности внешнего API.

Основные цели PR:

1. Создать таблицу `glossary_element` в PostgreSQL.
2. Реализовать получение элементов глоссария из внешнего API постранично через `limit` / `offset`.
3. Обеспечить полную перезапись данных, потому что уникальность внешних записей нельзя надёжно определить.
4. Минимизировать риск чтения неполного глоссария во время синхронизации.
5. Встроить новую логику в существующую архитектуру проекта:
   - `Settings` через `pydantic-settings`;
   - DI через `dishka`;
   - `UnitOfWork` + repository;
   - Alembic migration;
   - APScheduler;
   - логирование через updater logger.
6. Запускать синхронизацию по расписанию так же, как уже запускается обновление источников в `aisearch-updater`.

## 2. Краткий итог реализации

Добавлена новая вертикаль приложения:

```text
.env / Settings
    ↓
APScheduler в app.scheduler
    ↓
GlossaryService
    ↓
GlossaryAdapter → внешний API глоссария
    ↓
GlossaryRepository.replace_all(...)
    ↓
PostgreSQL: glossary_element + ge_splitted
```

Главный сценарий работы:

1. Scheduler запускает задачу `sync_glossary_*` по времени из `GLOSSARY_CRON_UPDATE_TIMES`.
2. `GlossaryService.sync_glossary()` начинает синхронизацию.
3. `GlossaryAdapter.fetch_all()` получает все элементы из внешнего API.
4. Только после полной успешной загрузки всех страниц сервис открывает `UnitOfWork`.
5. `GlossaryRepository.replace_all()` полностью заменяет содержимое `glossary_element`.
6. После вставки данных обновляется materialized view `ge_splitted`.
7. `UnitOfWork.commit()` фиксирует изменения.

Важно: если внешний API упал на середине загрузки, таблица в БД не трогается.

## 3. Изменённые файлы

В PR изменены и добавлены следующие файлы:

```text
.env.example
README.md
app/domain/repositories/glossary.py
app/domain/repositories/interfaces.py
app/domain/schemas/glossary.py
app/infrastructure/adapters/glossary.py
app/infrastructure/adapters/interfaces.py
app/infrastructure/ioc/search_ioc.py
app/infrastructure/migrations/base.py
app/infrastructure/migrations/versions/2026_04_24_1200-a3f9c1b7d2aa_glossary_element.py
app/infrastructure/models/glossary_element.py
app/infrastructure/unit_of_work/interfaces.py
app/infrastructure/unit_of_work/uow.py
app/scheduler.py
app/services/glossary.py
app/services/interfaces.py
app/settings/config.py
docker-compose.yml
tests/fixtures/settings.py
```

## 4. Настройки

Добавлен новый settings-класс:

```python
class GlossarySettings(EnvBaseSettings):
    ...
```

Префикс переменных окружения:

```text
GLOSSARY_
```

Добавленные переменные:

```env
GLOSSARY_API_URL="https://edu.emias.ru/edu-rest-api/test/glossary/glossary/getelements"
GLOSSARY_PAGE_LIMIT=500
GLOSSARY_CRON_UPDATE_TIMES="03:30"
GLOSSARY_REQUEST_TIMEOUT=30
GLOSSARY_MAX_RETRIES=5
GLOSSARY_RETRY_BACKOFF_BASE_SECONDS=1
GLOSSARY_RETRY_BACKOFF_MAX_SECONDS=30
GLOSSARY_ABBREVIATION_DELIMITER=";"
GLOSSARY_TERM_DELIMITER=";"
```

Назначение переменных:

| Переменная | Назначение |
|---|---|
| `GLOSSARY_API_URL` | URL внешнего API глоссария |
| `GLOSSARY_PAGE_LIMIT` | размер страницы при загрузке API, максимум 500 |
| `GLOSSARY_CRON_UPDATE_TIMES` | одно или несколько времён запуска синхронизации, формат `HH:MM` или `HH:MM,HH:MM` |
| `GLOSSARY_REQUEST_TIMEOUT` | timeout HTTP-запроса к API |
| `GLOSSARY_MAX_RETRIES` | максимальное число попыток запроса страницы |
| `GLOSSARY_RETRY_BACKOFF_BASE_SECONDS` | базовая задержка retry |
| `GLOSSARY_RETRY_BACKOFF_MAX_SECONDS` | максимальная задержка retry |
| `GLOSSARY_ABBREVIATION_DELIMITER` | delimiter для abbreviation на уровне настроек |
| `GLOSSARY_TERM_DELIMITER` | delimiter для term на уровне настроек |

Особенности:

- `page_limit` валидируется и должен быть в диапазоне от 1 до 500.
- `cron_update_times` валидируется общей функцией `_validate_cron_update_times`.
- Общая функция валидации времени теперь используется и для `ExtractEduSettings`, и для `GlossarySettings`.

Важный нюанс по delimiter:

- В settings есть `GLOSSARY_ABBREVIATION_DELIMITER` и `GLOSSARY_TERM_DELIMITER`.
- Но materialized view `ge_splitted` создаётся миграцией с фиксированным regexp `'; *'`.
- Если в будущем потребуется реально изменить delimiter для view, одного изменения `.env` будет недостаточно: потребуется новая миграция, пересоздающая `ge_splitted`.

## 5. Схема БД

Добавлена таблица:

```sql
CREATE TABLE glossary_element (
    id UUID PRIMARY KEY,
    abbreviation VARCHAR(500) NOT NULL DEFAULT '',
    term TEXT NOT NULL DEFAULT '',
    definition TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

Индексы:

```sql
CREATE INDEX ix_abbreviation_trgm
ON glossary_element USING gin (abbreviation gin_trgm_ops);

CREATE INDEX ix_term_trgm
ON glossary_element USING gin (term gin_trgm_ops);

CREATE INDEX ix_definition_trgm
ON glossary_element USING gin (definition gin_trgm_ops);
```

Назначение индексов:

- ускорить поиск по аббревиатуре;
- ускорить поиск по термину;
- ускорить поиск по определению;
- подготовить таблицу к будущему enrichment и lookup-сценариям.

Добавлена функция `modified_trigger()`, если её ещё нет:

```sql
CREATE OR REPLACE FUNCTION modified_trigger() RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.modified_at := NOW();
    RETURN NEW;
END;
$$;
```

Добавлен trigger на обновление `modified_at`:

```sql
CREATE TRIGGER modified_trigger
BEFORE UPDATE ON glossary_element
FOR EACH ROW EXECUTE FUNCTION modified_trigger();
```

Добавлена materialized view:

```sql
CREATE MATERIALIZED VIEW ge_splitted AS
SELECT *
FROM glossary_element
CROSS JOIN LATERAL
    regexp_split_to_table(glossary_element.abbreviation, '; *')
    AS abbreviation_splitted
CROSS JOIN LATERAL
    regexp_split_to_table(glossary_element.term, '; *')
    AS term_splitted;
```

Назначение `ge_splitted`:

- хранить “развёрнутые” варианты abbreviation и term;
- если в одном поле несколько значений через `;`, view позволяет работать с отдельными split-элементами;
- это пригодится для будущего enrichment и поиска совпадений.

Добавлен trigger для refresh materialized view:

```sql
CREATE TRIGGER trig_refresh_glossary
AFTER INSERT OR DELETE OR UPDATE ON glossary_element
FOR EACH STATEMENT EXECUTE FUNCTION refresh_glossary_view();
```

## 6. SQLAlchemy model

Добавлена модель:

```python
app/infrastructure/models/glossary_element.py
```

Модель:

```python
class GlossaryElement(Base):
    __tablename__ = "glossary_element"

    id: Mapped[uuid.UUID]
    abbreviation: Mapped[str]
    term: Mapped[str]
    definition: Mapped[str]
```

Модель наследуется от общего `Base`, поэтому получает поля:

- `created_at`;
- `modified_at`.

Модель зарегистрирована в:

```python
app/infrastructure/migrations/base.py
```

Это нужно, чтобы Alembic видел модель при autogenerate и чтобы структура проекта оставалась консистентной.

## 7. Domain schemas

Добавлены схемы:

```python
app/domain/schemas/glossary.py
```

Основные DTO:

```python
GlossaryElementCreateDTO
GlossaryElementSchema
```

Назначение:

- `GlossaryElementCreateDTO` используется при подготовке данных к массовой записи в БД;
- `GlossaryElementSchema` используется как response DTO репозитория.

Поля:

```python
id: uuid.UUID
abbreviation: str = ""
term: str = ""
definition: str = ""
created_at: datetime
modified_at: datetime
```

UUID создаётся в adapter при нормализации каждого элемента.

## 8. Adapter внешнего API

Добавлен adapter:

```python
app/infrastructure/adapters/glossary.py
```

Интерфейс:

```python
IGlossaryAdapter
```

Основной метод:

```python
async def fetch_all(self) -> list[GlossaryElementCreateDTO]:
    ...
```

Что делает adapter:

1. Берёт настройки из `settings.glossary`.
2. Делает GET-запросы к `GLOSSARY_API_URL`.
3. Передаёт параметры:
   - `limit`;
   - `offset`.
4. Получает ответ внешнего API.
5. Проверяет структуру ответа.
6. Нормализует элементы.
7. Повторяет запросы, пока не будут получены все данные.

Ожидаемый формат ответа API:

```json
{
  "data": [
    {
      "abbreviation": "...",
      "term": "...",
      "definition": "..."
    }
  ],
  "count": 1,
  "total": 100
}
```

Условия остановки пагинации:

- `total == 0`;
- `data` пустой;
- `count == 0`;
- количество накопленных элементов стало `>= total`.

Offset увеличивается на фактическую длину `data`, а не просто на `limit`.

Это важно, потому что API может вернуть меньше элементов, чем было запрошено.

### Retry и backoff

Retry реализован без новых зависимостей, на `asyncio` и `httpx`.

Retry выполняется для:

- `httpx.TimeoutException`;
- `httpx.ConnectError`;
- HTTP `429`;
- HTTP `5xx`.

Retry не выполняется для обычных `4xx`, кроме `429`.

Формула задержки:

```python
delay = min(base * 2 ** (attempt - 1), max_backoff)
```

Где:

- `base = GLOSSARY_RETRY_BACKOFF_BASE_SECONDS`;
- `max_backoff = GLOSSARY_RETRY_BACKOFF_MAX_SECONDS`.

### Нормализация элементов

Для каждого элемента:

- берутся только поля:
  - `abbreviation`;
  - `term`;
  - `definition`;
- `None` превращается в пустую строку;
- значения приводятся к строке;
- пробелы по краям удаляются;
- `abbreviation` обрезается до 500 символов, потому что в БД `VARCHAR(500)`;
- при обрезке пишется warning в лог;
- создаётся новый UUID.

Важно: adapter не пишет в БД. Он только загружает и нормализует данные.

## 9. Repository

Добавлен repository:

```python
app/domain/repositories/glossary.py
```

Интерфейс:

```python
IGlossaryRepository
```

Основной метод:

```python
async def replace_all(
    self,
    elements: list[GlossaryElementCreateDTO],
    batch_size: int = 5000,
) -> int:
    ...
```

Назначение `replace_all`:

- полностью заменить содержимое `glossary_element`;
- сделать это быстро;
- выполнить замену в рамках текущей транзакции SQLAlchemy session;
- обновить materialized view один раз после вставки всех данных.

### Защита от конкурентного запуска

Перед заменой берётся PostgreSQL advisory lock:

```sql
SELECT pg_try_advisory_lock(hashtext('glossary_sync'))
```

Если lock не получен, выбрасывается `ConflictError`.

В конце lock освобождается:

```sql
SELECT pg_advisory_unlock(hashtext('glossary_sync'))
```

Это защищает от ситуации, когда две синхронизации одновременно делают `TRUNCATE` / insert.

### Быстрая перезапись

Внутри `replace_all`:

1. Отключается trigger refresh materialized view:

   ```sql
   ALTER TABLE glossary_element DISABLE TRIGGER trig_refresh_glossary
   ```

2. Выполняется:

   ```sql
   TRUNCATE TABLE glossary_element
   ```

3. Данные вставляются batch insert через SQLAlchemy Core:

   ```python
   insert(GlossaryElement).values([...])
   ```

4. Выполняется один refresh:

   ```sql
   REFRESH MATERIALIZED VIEW ge_splitted
   ```

5. Trigger включается обратно:

   ```sql
   ALTER TABLE glossary_element ENABLE TRIGGER trig_refresh_glossary
   ```

Почему trigger отключается:

- если оставить trigger включённым, materialized view могла бы обновляться после каждого statement;
- при массовой вставке это лишняя нагрузка;
- быстрее и предсказуемее обновить view один раз после полной загрузки.

Важно:

- repository не делает `commit`;
- commit остаётся ответственностью `UnitOfWork` / service;
- при exception транзакция должна откатиться через существующий UoW-механизм.

## 10. Service

Добавлен сервис:

```python
app/services/glossary.py
```

Интерфейс:

```python
IGlossaryService
```

Основной метод:

```python
async def sync_glossary(self) -> int:
    ...
```

Что делает сервис:

1. Логирует начало синхронизации.
2. Вызывает `glossary_adapter.fetch_all()`.
3. После успешной загрузки всех элементов открывает `async with self.uow`.
4. Вызывает `self.uow.glossary.replace_all(elements=elements)`.
5. Делает `await self.uow.commit()`.
6. Логирует успешное завершение.
7. Возвращает количество записанных элементов.
8. При ошибке логирует exception и пробрасывает её выше.

Ключевой архитектурный принцип:

> Внешний API полностью читается до открытия транзакции замены данных.

Это уменьшает время lock-а в БД и исключает сценарий, когда таблица уже очищена, а API внезапно перестал отдавать следующие страницы.

## 11. UnitOfWork

Расширен `IUnitOfWork`:

```python
glossary: IGlossaryRepository
```

В `UnitOfWork.__aenter__` добавлено создание репозитория:

```python
self.glossary = GlossaryRepository(session=self.session)
```

Это делает работу с глоссарием консистентной с существующими репозиториями:

- `search_request`;
- `search_feedback`;
- `knowledge_feedback`.

## 12. DI

Обновлён provider:

```python
app/infrastructure/ioc/search_ioc.py
```

Добавлены зависимости:

```python
GlossaryAdapter → IGlossaryAdapter
GlossaryService → IGlossaryService
```

Это позволяет получать `IGlossaryService` из контейнера в scheduler.

## 13. Scheduler

Обновлён:

```python
app/scheduler.py
```

Добавлена функция:

```python
async def sync_glossary(service: IGlossaryService, logger: logging.Logger) -> None:
    ...
```

Scheduler теперь регистрирует отдельные cron jobs для глоссария:

```python
for time_str in settings.glossary.cron_update_times.split(","):
    hour, minute = map(int, time_str.strip().split(":"))
    task_name = f"sync_glossary_{hour:02d}_{minute:02d}"
    scheduler.add_job(...)
```

Сохраняются существующие `job_defaults`:

```python
{
    "misfire_grace_time": 300,
    "coalesce": True,
    "max_instances": 1,
}
```

Что это значит:

- если запуск был пропущен, есть 5 минут grace period;
- пропущенные выполнения объединяются;
- одновременно не запускается больше одного экземпляра одной scheduler-задачи.

Дополнительно repository использует advisory lock, поэтому защита есть и на уровне БД.

## 14. Docker Compose

В `docker-compose.yml` восстановлен базовый блок `aisearch-updater`.

Это важно, потому что `docker-compose.dev.yml` и `docker-compose.prod.yml` содержат override-части для `aisearch-updater`, но базовое описание сервиса должно существовать в `docker-compose.yml`.

Сервис продолжает запускать:

```bash
python3 -m app.scheduler
```

Новая синхронизация глоссария выполняется внутри того же scheduler-процесса.

Отдельный контейнер под глоссарий в PR не создавался.

### Почему синхронизация добавлена в `aisearch-updater`, а не отдельным сервисом

Плюсы текущего решения:

- не нужно плодить новый сервис;
- используется уже существующий scheduler;
- используется уже существующее логирование updater;
- меньше изменений в инфраструктуре;
- задача глоссария по природе похожа на другие scheduled update-задачи.

Минусы:

- `aisearch-updater` использует существующий search-образ, который может быть тяжелее, чем нужно для простой HTTP → PostgreSQL синхронизации;
- при падении updater не будут выполняться все scheduled-задачи, включая глоссарий;
- в будущем для независимого SLA может понадобиться отдельный lightweight-сервис.

Текущий вариант выбран как более простой и согласованный с архитектурой проекта.

## 15. Логирование

Логирование идёт через существующий `AISearchLogger` и updater logger.

Логируются:

- старт синхронизации;
- успешное завершение;
- количество загруженных/записанных элементов;
- ошибки API;
- retry attempts;
- mismatch между `count` и `len(data)`;
- обрезка слишком длинного `abbreviation`;
- exception при синхронизации.

## 16. Проверки

В PR выполнялись проверки:

```bash
python -m compileall app
```

Результат: успешно.

Также отдельно компилировались новые файлы миграции, адаптера и сервиса.

Миграции не были реально применены в окружении Codex из-за отсутствия Alembic:

```text
alembic: command not found
No module named alembic
```

Поэтому после merge обязательно желательно проверить в рабочем окружении:

```bash
alembic upgrade head
```

И затем вручную убедиться, что созданы:

- таблица `glossary_element`;
- индексы `ix_abbreviation_trgm`, `ix_term_trgm`, `ix_definition_trgm`;
- materialized view `ge_splitted`;
- trigger `modified_trigger`;
- trigger `trig_refresh_glossary`.

## 17. Что не реализовано в этом PR

Этот PR создаёт инфраструктуру синхронизации и хранения глоссария, но не реализует использование глоссария в поисковой логике.

Не сделано:

- enrichment пользовательского запроса;
- enrichment материалов перед индексацией;
- lookup-сервис для чтения глоссария;
- API-эндпоинты для просмотра глоссария;
- unit/integration tests для новой логики;
- отдельный lightweight-сервис под глоссарий;
- динамическое изменение delimiter для `ge_splitted` через `.env`.

## 18. Будущие точки развития

На базе этого PR можно дальше реализовать:

1. Query enrichment:
   - искать аббревиатуру в запросе пользователя;
   - добавлять term / definition / expanded synonyms;
   - использовать глоссарий до hybrid search.

2. Material enrichment:
   - обогащать вопросы/ответы/анализ при ingest;
   - добавлять расширенные поля в Milvus/OpenSearch;
   - индексировать варианты abbreviation/term.

3. Glossary lookup service:
   - отдельный read-only сервис поверх `glossary_element` или `ge_splitted`;
   - методы поиска по abbreviation, term, definition;
   - fuzzy/trigram search.

4. Integration tests:
   - adapter с mock HTTP API;
   - repository на test Postgres;
   - scheduler job registration;
   - миграция `upgrade/downgrade`.

5. Отдельный сервис:
   - если синхронизация глоссария должна жить независимо от `aisearch-updater`;
   - если нужен lightweight image без search/GPU-зависимостей;
   - если понадобится отдельный мониторинг/SLA.

## 19. Важные архитектурные договорённости для будущих задач

1. Не читать внешний API глоссария в runtime-поиске.
   Runtime-поиск должен использовать локальную БД или кеш, иначе внешний API станет точкой отказа поиска.

2. Не делать частичный upsert по внешним данным, пока нет надёжного уникального ключа.
   Сейчас выбран full replace, потому что уникальность элементов глоссария нельзя гарантировать.

3. Сначала полностью загрузить API, потом открывать транзакцию замены.
   Это уменьшает время блокировки таблицы.

4. Не переносить `commit` в repository.
   В проекте commit — ответственность UnitOfWork/service.

5. Не менять delimiter для materialized view только через `.env`.
   Если delimiter должен поменяться для split view, нужна новая миграция.

6. Если добавляется чтение глоссария для enrichment, лучше завести отдельный read repository/service, а не расширять adapter.
   Adapter отвечает за внешний API, repository/service — за локальные данные.

7. Если enrichment будет использоваться в поиске, нужно отдельно продумать кеширование.
   Частые обращения к PostgreSQL на каждый поисковый запрос могут стать узким местом.

## 20. Краткая карта компонентов

| Компонент | Файл | Ответственность |
|---|---|---|
| Settings | `app/settings/config.py` | env-настройки синхронизации |
| Model | `app/infrastructure/models/glossary_element.py` | SQLAlchemy-модель таблицы |
| Migration | `app/infrastructure/migrations/versions/2026_04_24_1200-a3f9c1b7d2aa_glossary_element.py` | создание таблицы, индексов, view и triggers |
| DTO | `app/domain/schemas/glossary.py` | схемы данных глоссария |
| Adapter | `app/infrastructure/adapters/glossary.py` | загрузка и нормализация данных из API |
| Repository | `app/domain/repositories/glossary.py` | атомарная полная перезапись таблицы |
| Service | `app/services/glossary.py` | orchestration API fetch → DB replace |
| UoW | `app/infrastructure/unit_of_work/uow.py` | прокидывание glossary repository |
| DI | `app/infrastructure/ioc/search_ioc.py` | регистрация adapter/service |
| Scheduler | `app/scheduler.py` | запуск sync по расписанию |
| Compose | `docker-compose.yml` | запуск `aisearch-updater` |
| Env docs | `.env.example`, `README.md` | документация переменных |

## 21. Мини-чеклист для следующей LLM-задачи

Если следующая задача будет про использование глоссария в поиске, сначала нужно определить:

- где будет выполняться enrichment: до очереди, внутри worker, внутри orchestrator или на этапе ingest;
- нужно ли enrichment применять к query, к материалам или к обоим;
- какие поля использовать: `abbreviation`, `term`, `definition`, `ge_splitted.abbreviation_splitted`, `ge_splitted.term_splitted`;
- нужна ли точная нормализация регистра/пробелов/пунктуации;
- нужно ли кешировать глоссарий в памяти/Redis;
- как измерять влияние enrichment на качество выдачи.
