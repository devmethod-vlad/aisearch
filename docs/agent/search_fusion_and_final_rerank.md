# Hybrid search: разделение fusion и final rank

Дата актуализации: 2026-05-14  
Контекст: `devmethod-vlad/aisearch`, актуальная модель гибридного pipeline.

## 1. Зачем разделены retrieval branches, fusion и final rank

Новая модель явно разделяет три этапа, чтобы упростить настройку и диагностику:

1. **Retrieval branches** — dense и lexical ветки независимо находят кандидатов.
2. **Fusion** — объединение dense/lex сигналов в единый `score_fusion`.
3. **Final rank** — финальная сортировка кандидатов (с CE или без CE), управляемая отдельным режимом.

Ключевая идея: fusion отвечает только за retrieval-сигналы dense/lex, а final rank отдельно решает, как использовать cross-encoder.

---

## 2. `HYBRID_FUSION_MODE`: как объединяются dense/lex сигналы

Допустимые значения:

```env
HYBRID_FUSION_MODE=weighted_score
HYBRID_FUSION_MODE=rrf
```

### 2.1 `weighted_score`

Используется взвешенная сумма retrieval-оценок:

```text
score_fusion = w_dense * score_dense + w_lex * score_lex
```

Где:

- `w_dense` = `HYBRID_W_DENSE`
- `w_lex` = `HYBRID_W_LEX`

### 2.2 `rrf`

Используется Reciprocal Rank Fusion по позициям документа в dense и lexical списках:

```text
contribution_dense = rrf_w_dense / (rrf_k + dense_rank)
contribution_lex = rrf_w_lex / (rrf_k + lex_rank)
score_rrf_raw = contribution_dense + contribution_lex
score_fusion = normalized RRF score или основной RRF fusion score, согласно реализации
```

Где:

- `rrf_k` = `HYBRID_RRF_K`
- `rrf_w_dense` = `HYBRID_RRF_W_DENSE`
- `rrf_w_lex` = `HYBRID_RRF_W_LEX`

---

## 3. `HYBRID_FINAL_RANK_MODE`: как формируется финальная сортировка

Допустимые значения:

```env
HYBRID_FINAL_RANK_MODE=fusion_only
HYBRID_FINAL_RANK_MODE=ce_final
HYBRID_FINAL_RANK_MODE=ce_blend
HYBRID_FINAL_RANK_MODE=legacy_weighted
```

### 3.1 `fusion_only`

- CE не запускается.
- Финальный score:

```text
score_final = score_fusion
```

### 3.2 `ce_final`

- CE запускается.
- Финальный score:

```text
score_final = score_ce_raw
```

- Сортировка:

```text
sort: score_ce_raw desc, score_fusion desc
```

- `HYBRID_FINAL_CE_SCORE` в этом режиме не используется.

### 3.3 `ce_blend`

- CE запускается только если `HYBRID_FINAL_W_CE > 0`.
- Финальный score строится из нормализованного fusion и нормализованного CE:

```text
score_final = normalized score_fusion + normalized CE
```

- Используются:
  - `HYBRID_FINAL_W_FUSION`
  - `HYBRID_FINAL_W_CE`
  - `HYBRID_FINAL_FUSION_NORM`
  - `HYBRID_FINAL_CE_SCORE`

### 3.4 `legacy_weighted`

- Разрешён только с `HYBRID_FUSION_MODE=weighted_score`.
- Сохраняет старую удачную формулу:

```text
score_final = w_dense * score_dense_norm + w_lex * score_lex + final_w_ce * score_ce
```

- Используется `score_ce`, а не `score_ce_raw`.
- `HYBRID_FINAL_W_CE` заменяет прежний `HYBRID_W_CE`.

---

## 4. Когда запускается cross-encoder

- `fusion_only`: **никогда**.
- `ce_final`: **всегда**, если есть candidates.
- `ce_blend`: **только если** `HYBRID_FINAL_W_CE > 0`.
- `legacy_weighted`: **только если** `HYBRID_FINAL_W_CE > 0`.

---

## 5. Когда запускается OpenSearch (lexical branch)

OpenSearch branch не запускается, если выполняется хотя бы одно условие:

- `SEARCH_USE_OPENSEARCH=false`
- `lex_top_k <= 0`
- `HYBRID_FUSION_MODE=weighted_score` и `w_lex <= 0`
- `HYBRID_FUSION_MODE=rrf` и `rrf_w_lex <= 0`

---

## 6. Short mode override

Для short-режима используются отдельные override-параметры:

- `SHORT_FUSION_MODE`
- `SHORT_DENSE_TOP_K`
- `SHORT_LEX_TOP_K`
- `SHORT_TOP_K`
- `SHORT_W_DENSE`
- `SHORT_W_LEX`
- `SHORT_RRF_K`
- `SHORT_RRF_W_DENSE`
- `SHORT_RRF_W_LEX`
- `SHORT_FINAL_W_FUSION`
- `SHORT_FINAL_W_CE`
- `SHORT_FINAL_FUSION_NORM`
- `SHORT_FINAL_CE_SCORE`
- `SHORT_USE_OPENSEARCH`

Важно: `SHORT_FINAL_RANK_MODE` **не вводится**. Режим `HYBRID_FINAL_RANK_MODE` остаётся глобальным.

---

## 7. Score fields

- `score_dense` — dense score после внутренней нормализации.
- `score_dense_raw` — исходный dense score до нормализации.
- `score_lex` — lexical score (нормализованный внутри lexical-ветки).
- `score_fusion` — итоговый fusion score после объединения dense/lex.
- `score_rrf_raw` — raw RRF score до нормализации (актуален для `rrf`).
- `score_fusion_norm` — нормализованный `score_fusion`, применяемый в blend-логике.
- `score_ce_raw` — raw логит(ы) cross-encoder.
- `score_ce` — CE score в рабочей шкале режима, где он нужен.
- `score_ce_norm` — нормализованный CE score для blend-логики.
- `score_final` — финальный score, по которому формируется выдача.
- `score_final_mode` — режим формирования `score_final` (значение `HYBRID_FINAL_RANK_MODE`).

---

## 8. Актуальные env-примеры

### A) Production baseline

```env
HYBRID_FUSION_MODE=weighted_score
HYBRID_FINAL_RANK_MODE=legacy_weighted
HYBRID_W_DENSE=0.55
HYBRID_W_LEX=0.15
HYBRID_FINAL_W_CE=0.3
```

### B) `weighted_score` без CE

```env
HYBRID_FUSION_MODE=weighted_score
HYBRID_FINAL_RANK_MODE=fusion_only
```

### C) `rrf` без CE

```env
HYBRID_FUSION_MODE=rrf
HYBRID_FINAL_RANK_MODE=fusion_only
```

### D) `rrf + ce_blend`

```env
HYBRID_FUSION_MODE=rrf
HYBRID_FINAL_RANK_MODE=ce_blend
HYBRID_RRF_W_DENSE=1.0
HYBRID_RRF_W_LEX=2.0
HYBRID_FINAL_W_FUSION=0.7
HYBRID_FINAL_W_CE=0.3
```

### E) `ce_final`

```env
HYBRID_FINAL_RANK_MODE=ce_final
```

---

## 9. Инварианты

1. Fusion объединяет только dense/lex retrieval-сигналы.
2. Final rank mode определяет, используется ли CE и как.
3. CE не является всегда финальным сортировщиком.
4. `legacy_weighted` сохраняет старую удачную weighted формулу.
5. `legacy_weighted` разрешён только с `weighted_score`.
6. `fusion_only` не запускает CE.
7. `ce_final` использует raw CE logits.
8. `ce_blend` использует `HYBRID_FINAL_CE_SCORE`.
9. `SEARCH_USE_OPENSEARCH` остаётся отдельным переключателем lexical branch.
