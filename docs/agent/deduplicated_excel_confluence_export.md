# Экспорт дедуплицированных знаний в Excel и загрузка в Confluence

## 1) Для чего добавлена логика

Добавлена служебная выгрузка дедуплицированного набора знаний из `UpdaterService.update_all` в Excel с последующей загрузкой в Confluence как attachment. Это нужно для прозрачности качества данных и быстрого контроля результата объединения источников KB/ВиО после дедупликации.

## 2) Где логика встроена в пайплайн updater

Точка интеграции — `app/services/updater.py`, метод `UpdaterService.update_all`.

Экспорт вызывается в конце `update_all`, после того как:

1. данные дедуплицированы (`dedup_by_question_any`);
2. разделены по источникам (`split_by_source`);
3. для каждой части выполнен `prepare_dataframe`;
4. каждая подготовленная часть отправлена в `_update_collection_from_df` (сначала ТП, затем ВиО).

Только после этого объединяется `prepared`-результат двух частей и формируется Excel.

## 3) Почему экспорт выполняется после `prepare_dataframe` и upsert-шагов

Экспорт должен отражать фактические данные, которые прошли ingestion-фильтры и реально участвуют в обновлении Milvus/OpenSearch. Поэтому Excel строится по объединенному `prepared`-датафрейму (ТП + ВиО после фильтрации), а не по промежуточному набору до `prepare_dataframe`.

При этом служебные поля (`row_idx`, token-поля) исключаются из листа `Знания` на этапе экспорта.

## 5) Структура Excel

Формируется in-memory `.xlsx` с двумя листами:

### Лист `Знания`

- содержит дедуплицированные записи;
- колонки переименовываются обратно во внешние русские названия;
- обратный маппинг строится из `field_mapping.json` (`russian_name -> internal_name` => `internal_name -> russian_name`);
- порядок колонок следует порядку ключей JSON-маппинга;
- колонки вне маппинга добавляются в конец.

### Лист `Статистика`

- строится сравнением двух срезов:
  1) **до фильтрации**: сразу после `combine_validated_sources` и строго до `dedup_by_question_any`;
  2) **после фильтрации**: по объединённому `prepared`-датафрейму после `prepare_dataframe` для ТП и ВиО;
- агрегат для обоих срезов: количество уникальных `ext_id` по источнику;
- итоговые колонки листа:
  - `Источник`
  - `Всего (до фильтрации)`
  - `Всего (после фильтрации)`

## 6) Добавленные env-переменные

В `ExtractEduSettings` добавлены настройки с префиксом `EXTRACT_`:

- `EXTRACT_DEDUPLICATED_EXCEL_UPLOAD_ENABLED` (bool, default `false`)
- `EXTRACT_DEDUPLICATED_EXCEL_FILE_NAME_TEMPLATE` (default `statistic_{timestamp}.xlsx`)
- `EXTRACT_DEDUPLICATED_EXCEL_KEEP_VERSIONS` (int, default `5`)
- `EXTRACT_DEDUPLICATED_EXCEL_MAX_RETRIES` (int, default `5`)
- `EXTRACT_DEDUPLICATED_EXCEL_RETRY_BACKOFF_BASE_SECONDS` (float, default `1`)
- `EXTRACT_DEDUPLICATED_EXCEL_RETRY_BACKOFF_MAX_SECONDS` (float, default `30`)

Валидации:

- `keep_versions >= 1`
- `max_retries >= 1`
- backoff-значения неотрицательные

## 7) Как работает загрузка в Confluence

Реализовано в `app/infrastructure/adapters/edu.py`:

- поиск attachment на странице по имени (с fallback-фильтрацией по `title`);
- если файл не найден — создаётся новый attachment;
- если найден — загружается новая версия существующего attachment;
- после успешной загрузки запускается очистка старых версий.

Используется Confluence REST API Data Center:

- `GET /rest/api/content/{page_id}/child/attachment`
- `POST /rest/api/content/{page_id}/child/attachment`
- `POST /rest/api/content/{page_id}/child/attachment/{attachment_id}/data`
- `GET /rest/api/content/{attachment_id}/version`
- `DELETE /rest/api/content/{attachment_id}/version/{version_number}`

## 8) Как работает создание новой версии attachment

Метод `upload_or_update_attachment_to_edu`:

1. ищет attachment по `filename`;
2. при наличии `attachment_id` отправляет upload в endpoint `.../{attachment_id}/data`;
3. при отсутствии создаёт новый attachment;
4. логирует выбранный путь (новый файл или новая версия).

## 9) Как работает ограничение количества версий

После успешной загрузки:

1. читаются все версии attachment;
2. версии сортируются по номеру;
3. если версий больше `keep_last_versions`, удаляются самые старые;
4. текущая/последняя версия не затрагивается.

Ошибки удаления конкретной старой версии логируются отдельно и не прерывают основной поток.

## 10) Почему ошибки экспорта не валят основной `update_all`

В `update_all` экспорт обёрнут в `try/except`:

- любая ошибка формирования Excel или загрузки attachment логируется через `logger.exception`;
- основной процесс split/prepare/upsert в Milvus и OpenSearch продолжается без изменений.

## 11) Изменённые файлы

- `app/infrastructure/utils/deduplicated_excel_export.py` (новый)
- `app/settings/config.py`
- `.env.example`
- `app/infrastructure/adapters/interfaces.py`
- `app/infrastructure/adapters/edu.py`
- `app/services/updater.py`
- `tests/mocks/edu.py`
- `tests/unit/test_deduplicated_excel_export.py` (новый)
- `tests/unit/test_edu_confluence_upload.py` (новый)
- `tests/unit/test_updater_deduplicated_export.py` (новый)
- `docs/agent/deduplicated_excel_confluence_export.md` (новый)

## 12) Как тестировать

Рекомендуемый минимум:

1. `ruff check .`
2. `pytest tests/unit/test_deduplicated_excel_export.py`
3. `pytest tests/unit/test_edu_confluence_upload.py`
4. `pytest tests/unit/test_updater_deduplicated_export.py`

Полный прогон:

- `pytest`

Если интеграционные контейнерные тесты недоступны в окружении, достаточно запускать unit-тесты из списка выше.
