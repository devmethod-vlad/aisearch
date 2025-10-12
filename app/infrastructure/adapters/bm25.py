import pickle
import typing as tp
import time

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from app.common.logger import AISearchLogger
from app.infrastructure.adapters.interfaces import IBM25Adapter
from app.infrastructure.utils.nlp import normalize_query
from app.infrastructure.utils.metrics import metrics_print
from app.settings.config import Settings


class BM25Adapter(IBM25Adapter):
    """Адаптер bm25 на rank_bm25"""

    def __init__(self, settings: Settings, logger: AISearchLogger):
        bm25_init_start = time.perf_counter()
        self.index_path = settings.bm25.index_path
        self.schema_fields = settings.bm25.schema_fields
        self._ix: BM25Okapi | None = None
        self._data: pd.DataFrame | None = None
        self.ensure_index() if settings.switches.use_bm25 else None
        self.logger = logger
        metrics_print("🕒 Инициализация BM25", bm25_init_start)


    @staticmethod
    def build_index(
        data: pd.DataFrame, index_path: str, texts: list[str], logger: AISearchLogger
    ) -> None:
        """Построение индекса"""
        logger.info("Построение индекса BM25 ...")

        tokenized_corpus = [normalize_query(t) for t in texts]

        with open(f"{index_path}/bm25.pkl", "wb") as f:
            pickle.dump(
                {
                    "bm25": BM25Okapi(tokenized_corpus),
                    "tokenized_corpus": tokenized_corpus,
                },
                f,
            )

        data.to_parquet(f"{index_path}/rows.parquet")
        logger.info("Индекс BM25 построен")

    def ensure_index(self) -> None:
        """Подгрузка индекса"""
        try:
            with open(f"{self.index_path}/bm25.pkl", "rb") as f:
                bm25_pack = pickle.load(f)
            self._ix = bm25_pack["bm25"]
            self._data = pd.read_parquet(f"{self.index_path}/rows.parquet")
        except Exception as e:
            self.logger.warning(f"Не удалось подгрузить индексы из {self.index_path}, error: {e}")

    def search(self, query: str, top_k: int = 50) -> list[dict[str, tp.Any]]:
        """Возвращает список кандидатов:
        [{"ext_id","question","analysis","answer","score_bm25","source":"bm25"}, ...]
        """
        scores = self._ix.get_scores(query)
        idxs = np.argsort(scores)[::-1][:top_k]
        hits = [(int(i), float(scores[i])) for i in idxs]
        results: list[dict[str, tp.Any]] = []
        seen = set()

        data_columns = list(self._data.columns)

        for ridx, s in hits:
            if ridx in seen:
                continue
            seen.add(ridx)
            r = self._data.loc[ridx]

            result_item = {}
            for col in data_columns:
                result_item[col] = str(r.get(col, "") if hasattr(r, "get") else getattr(r, col, ""))

            result_item["score_bm25"] = float(s)  # type: ignore
            result_item["source"] = "bm25"

            results.append(result_item)

            if len(results) >= top_k:
                break

        return results
