# Hybrid search: fusion dense/lex и финальный rerank cross-encoder

Дата актуализации: 2026-05-07  
Контекст: `devmethod-vlad/aisearch`, изменения гибридного поиска после PR #15.

## 1. Цель изменений

Изменения нужны, чтобы разделить два разных уровня гибридного поиска:

1. **Retrieval** — dense и lexical ветки находят кандидатов.
2. **Rerank** — cross-encoder точнее сортирует уже найденные кандидаты.

Раньше итоговый score мог собираться как сумма трёх компонентов:

```text
score_final = w_dense * score_dense + w_lex * score_lex + w_ce * score_ce
```

Такая схема смешивает retrieval-сигналы и rerank-сигнал в одной формуле. Новая модель проще:

```text
dense + lex -> fusion -> score_fusion -> optional cross-encoder -> score_final
```

Cross-encoder, если включён, теперь всегда является финальным сортировщиком.

## 2. Что добавлено

### `HYBRID_FUSION_MODE`

Переменная управляет способом объединения dense/lex кандидатов.

Допустимые значения:

```env
HYBRID_FUSION_MODE=weighted_score
HYBRID_FUSION_MODE=rrf
```

- `weighted_score` — взвешенная сумма `score_dense` и `score_lex`.
- `rrf` — Reciprocal Rank Fusion по позициям документов в dense/lex списках.

### `HYBRID_RRF_K`

Параметр RRF:

```env
HYBRID_RRF_K=60
```

Используется только при:

```env
HYBRID_FUSION_MODE=rrf
```

Чем больше `rrf_k`, тем мягче влияние разницы между соседними позициями.

## 3. Что не добавлено

Не вводится `HYBRID_RERANK_MODE`.

Рассматривались два варианта:

- `equal` — cross-encoder как равноправный компонент score;
- `final` — cross-encoder как финальный сортировщик.

От варианта `equal` отказались, чтобы не усложнять pipeline. Итоговое правило одно:

```text
если cross-encoder включён, он финально сортирует результаты;
если выключен, результаты сортируются по score_fusion.
```

## 4. Режим `weighted_score`

Используется при:

```env
HYBRID_FUSION_MODE=weighted_score
```

Формула retrieval fusion:

```text
score_fusion = w_dense * score_dense + w_lex * score_lex
```

Где:

- `w_dense` — `HYBRID_W_DENSE`;
- `w_lex` — `HYBRID_W_LEX`.

`HYBRID_W_CE` больше не должен участвовать в этой формуле.

Если cross-encoder выключен:

```text
score_final = score_fusion
sort: score_fusion desc
```

Если cross-encoder включён:

```text
score_final = score_ce
sort: score_ce desc, score_fusion desc
```

`score_fusion` остаётся полезным как retrieval score и tie-breaker.

## 5. Режим `rrf`

Используется при:

```env
HYBRID_FUSION_MODE=rrf
```

RRF объединяет не абсолютные score, а ранги документов в dense/lex списках.

Для dense-вклада:

```text
contribution_dense = w_dense / (rrf_k + dense_rank)
```

Для lexical-вклада:

```text
contribution_lex = w_lex / (rrf_k + lex_rank)
```

Raw score:

```text
score_rrf_raw = contribution_dense + contribution_lex
```

Если документ найден обеими ветками, он получает вклад от обеих. Это поднимает документы, которые одновременно хорошо находятся semantic и lexical поиском.

После расчёта raw score выполняется нормализация в рамках одного запроса:

```text
score_fusion = score_rrf_raw / max(score_rrf_raw)
```

Если максимум равен нулю, `score_fusion = 0.0`.

Если cross-encoder выключен:

```text
score_final = score_fusion
sort: score_fusion desc
```

Если cross-encoder включён:

```text
score_final = score_ce
sort: score_ce desc, score_fusion desc
```

Важно: cross-encoder не добавляется внутрь RRF как третий источник. RRF объединяет только retrieval-ветки dense и lex.

## 6. Роль весов

### `HYBRID_W_DENSE`

В `weighted_score`:

```text
score_fusion += HYBRID_W_DENSE * score_dense
```

В `rrf`:

```text
score_rrf_raw += HYBRID_W_DENSE / (rrf_k + dense_rank)
```

### `HYBRID_W_LEX`

В `weighted_score`:

```text
score_fusion += HYBRID_W_LEX * score_lex
```

В `rrf`:

```text
score_rrf_raw += HYBRID_W_LEX / (rrf_k + lex_rank)
```

### `HYBRID_W_CE`

`HYBRID_W_CE` больше не является весом в итоговой формуле.

Рекомендуемая новая семантика:

```text
SEARCH_USE_RERANKER=true и HYBRID_W_CE > 0.0 -> cross-encoder запускается
SEARCH_USE_RERANKER=false или HYBRID_W_CE == 0.0 -> cross-encoder не запускается
```

Если cross-encoder запущен, он финально сортирует результаты независимо от численного значения `HYBRID_W_CE`.

## 7. Поля score в результате

### `score_dense`

Score dense retrieval после внутренней нормализации.

### `score_lex`

Score lexical retrieval после нормализации OpenSearch score относительно top lexical hit.

### `score_fusion`

Главный retrieval score после объединения dense/lex.

- В `weighted_score`: weighted dense/lex score.
- В `rrf`: нормализованный RRF score.

### `score_rrf_raw`

Raw RRF score. Присутствует только при `HYBRID_FUSION_MODE=rrf`.

### `score_ce`

Score cross-encoder. Присутствует, если cross-encoder реально запускался.

### `score_final`

Score, по которому сформирована финальная выдача.

```text
CE включён: score_final = score_ce
CE выключен: score_final = score_fusion
```

## 8. Cache key

Режим fusion влияет на выдачу, поэтому должен участвовать в cache key.

Минимальная модель:

```text
hybrid_version = {HYBRID_VERSION}:fusion={HYBRID_FUSION_MODE}:rrf_k={HYBRID_RRF_K}
```

Это нужно, чтобы не смешивать кеши для:

- `weighted_score`;
- `rrf`;
- разных значений `rrf_k`.

Существующие части cache key должны сохраниться:

- query hash;
- `top_k`;
- data version;
- collection/index;
- presearch key part;
- token/exact filters key part.

## 9. Intermediate results

При `show_intermediate_results=true` полезно видеть все основные стадии pipeline.

Сохраняются ключи:

```text
dense
lex
ce
```

Добавляется ключ:

```text
fusion
```

Смысл:

- `dense` — top кандидаты по `score_dense`;
- `lex` — top кандидаты по `score_lex`;
- `fusion` — top merged candidates по `score_fusion`;
- `ce` — top reranked candidates по `score_ce`, если cross-encoder запускался.

## 10. Инварианты для дальнейшей разработки

1. Fusion отвечает только за объединение dense/lex.
2. Cross-encoder не участвует в RRF.
3. Cross-encoder всегда финальный сортировщик, если включён.
4. Старую формулу с `w_ce * score_ce` в `score_final` не возвращать.
5. При выключенном cross-encoder выдача сортируется по `score_fusion`.
6. `weighted_score` не удалять: это режим обратной совместимости.
7. Request-driven runtime из PR #15 не откатывать.
8. Не возвращать глобальные env-флаги `APP_USE_CACHE`, `HYBRID_PRESEARCH_ENABLED`, `HYBRID_PRESEARCH_FIELD`, `HYBRID_ENABLE_INTERMEDIATE_RESULTS`.

## 11. Рекомендуемые env-примеры

Базовый weighted режим:

```env
HYBRID_FUSION_MODE=weighted_score
HYBRID_RRF_K=60
SEARCH_USE_RERANKER=true
HYBRID_W_CE=0.3
```

RRF с cross-encoder:

```env
HYBRID_FUSION_MODE=rrf
HYBRID_RRF_K=60
SEARCH_USE_RERANKER=true
HYBRID_W_CE=0.3
```

RRF без cross-encoder:

```env
HYBRID_FUSION_MODE=rrf
HYBRID_RRF_K=60
SEARCH_USE_RERANKER=false
```

## 12. Что проверять тестами

Минимальные тестовые сценарии:

1. `weighted_score` без CE:
   - `score_fusion = w_dense * score_dense + w_lex * score_lex`;
   - `score_final = score_fusion`.

2. `weighted_score` с CE:
   - сортировка по `score_ce`;
   - `score_fusion` используется как tie-breaker;
   - `score_final = score_ce`.

3. RRF merge:
   - dense/lex объединяются по `merge_by_field`;
   - документ из обеих веток получает оба вклада;
   - есть `score_rrf_raw`;
   - `score_fusion` нормализован в `[0, 1]`.

4. `rrf` без CE:
   - сортировка по `score_fusion`;
   - `score_final = score_fusion`.

5. `rrf` с CE:
   - сортировка по `score_ce`;
   - `score_fusion` используется как tie-breaker;
   - `score_final = score_ce`.

6. Cache key:
   - различается для `weighted_score` и `rrf`;
   - различается для разных `HYBRID_RRF_K`.

7. Intermediate results:
   - при `show_intermediate_results=true` есть `intermediate_results["fusion"]`.

## 13. Краткая схема pipeline

```text
1. Read Redis pack.
2. Parse query, top_k, runtime flags, presearch, filters.
3. Normalize token/exact filters.
4. Build cache key with fusion mode and rrf_k.
5. Try cache if allowed.
6. Optional presearch.
7. Encode query.
8. Run dense and lexical branches in parallel.
9. Apply dense/lex precut.
10. Apply fusion: weighted_score or rrf.
11. Get merged candidates with score_fusion.
12. If CE enabled, calculate score_ce.
13. Final sort:
    - CE enabled: score_ce desc, score_fusion desc;
    - CE disabled: score_fusion desc.
14. Set score_final.
15. Inject presearch result if present.
16. Save cache/result/metrics.
```

## 14. Практический смысл

После изменений pipeline становится проще анализировать:

```text
retrieval отвечает за recall;
fusion объединяет retrieval-источники;
cross-encoder отвечает за precision финальной выдачи.
```

Если хорошие документы не попали в candidates — проблема в dense/lex retrieval или precut.  
Если документы есть в `fusion`, но порядок плохой — смотреть cross-encoder.  
Если weighted fusion нестабилен из-за разных шкал dense/lex — включать `HYBRID_FUSION_MODE=rrf`.
