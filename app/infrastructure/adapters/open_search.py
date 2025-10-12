import typing as tp
import time

import pandas as pd
from opensearchpy import (
    OpenSearch,
    helpers as os_helpers,
)

from app.common.logger import AISearchLogger
from app.infrastructure.adapters.interfaces import IOpenSearchAdapter
from app.infrastructure.utils.metrics import metrics_print
from app.settings.config import Settings


class OpenSearchAdapter(IOpenSearchAdapter):
    """Адаптер для opensearch"""

    def __init__(self, settings: Settings, logger: AISearchLogger):
        os_init_start = time.perf_counter()
        self.config = settings.opensearch
        self.client = OpenSearch(
            hosts=[{"host": self.config.host, "port": self.config.port}],
            http_compress=True,
            http_auth=((self.config.user, self.config.password) if self.config.user else None),
            use_ssl=self.config.use_ssl,
            verify_certs=self.config.verify_certs,
        )
        self.logger = logger
        metrics_print("🕒 Инициализация OPENSEARCH", os_init_start)


    def build_index(self, data: pd.DataFrame) -> None:
        """Построение индекса"""
        self.logger.info("Построение индекса OS ...")
        self._ensure_os_index(recreate=self.config.recreate_index)
        self._os_optimize_for_bulk(on=True)
        self._os_bulk_index(data=data, chunk_size=self.config.bulk_chunk_size)
        self._os_optimize_for_bulk(on=False)
        self.logger.info("Индекс OS построен")

    def search(self, body: dict, size: int) -> list[dict]:
        """Поиск при помощи opensearch"""
        resp = self.client.search(index=self.config.index_name, body=body, size=size)
        return resp["hits"]["hits"]

    def _os_optimize_for_bulk(self, on: bool) -> None:
        if on:
            self.client.indices.put_settings(
                index=self.config.index_name,
                body={"index": {"refresh_interval": "-1", "translog.durability": "async"}},
            )
        else:
            self.client.indices.put_settings(
                index=self.config.index_name,
                body={"index": {"refresh_interval": "1s", "translog.durability": "request"}},
            )
            self.client.indices.refresh(index=self.config.index_name)

    def _os_bulk_index(self, data: pd.DataFrame, chunk_size: int = 1000) -> None:
        rows = []
        for _, r in data.iterrows():
            row_dict = {col: str(r[col]) for col in data.columns}
            rows.append(row_dict)

        scalar_keys = list(rows[0].keys()) if rows else []

        def gen_actions() -> tp.Generator[dict, None, None]:
            for r in rows:
                doc = {key: r[key] for key in scalar_keys}
                yield {
                    "_op_type": "index",
                    "_index": self.config.index_name,
                    "_id": r["row_idx"],
                    "_source": doc,
                }

        os_helpers.bulk(self.client, gen_actions(), chunk_size=chunk_size, request_timeout=120)

    def _ensure_os_index(self, recreate: bool = False) -> None:
        """Создание индекса с русским анализатором.
        OS_INDEX_ANSWER=true|false — индексировать ли поле answer как text (иначе останется в _source).
        """
        if recreate and self.client.indices.exists(
            index=self.config.index_name, expand_wildcards="all"
        ):
            self.client.indices.delete(index=self.config.index_name, ignore=[400, 404])

        if not self.client.indices.exists(index=self.config.index_name):
            # Русский анализатор + базовый мэппинг
            answer_mapping = (
                {"type": "text", "analyzer": "ru_mixed"}
                if self.config.index_answer
                else {"type": "text", "index": False}  # type: ignore
            )

            body = {
                "settings": {
                    "index": {"number_of_shards": 1, "number_of_replicas": 0},
                    "analysis": {
                        "filter": {
                            "ru_stop": {"type": "stop", "stopwords": "_russian_"},
                            "ru_stemmer": {"type": "stemmer", "language": "russian"},
                        },
                        "analyzer": {
                            "ru_mixed": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "char_filter": ["html_strip"],
                                "filter": ["lowercase", "ru_stop", "ru_stemmer"],
                            }
                        },
                    },
                },
                "mappings": {
                    "properties": {
                        "row_idx": {"type": "integer"},
                        "source": {"type": "keyword"},
                        "ext_id": {"type": "keyword"},
                        "page_id": {"type": "keyword"},
                        "role": {"type": "keyword"},
                        "component": {"type": "keyword"},
                        "question": {
                            "type": "text",
                            "analyzer": "ru_mixed",
                            "fields": {"kw": {"type": "keyword", "ignore_above": 1024}},
                        },
                        "analysis": {"type": "text", "analyzer": "ru_mixed"},
                        "answer": answer_mapping,
                    }
                },
            }
            self.client.indices.create(index=self.config.index_name, body=body)
