import json
import time
import typing as tp

import pandas as pd
from opensearchpy import (
    OpenSearch,
    helpers as os_helpers,
)

from app.common.logger import AISearchLogger
from app.infrastructure.adapters.interfaces import IOpenSearchAdapter
from app.infrastructure.utils.metrics import metrics_print
from app.domain.schemas.open_search import OSIndexSchema
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
        self.os_schema = self._load_os_schema_from_json(self.config.schema_path)
        metrics_print("🕒 Инициализация OPENSEARCH", os_init_start)



    def build_index(self, data: dict[str, tp.Any]) -> None:
        """Построение индекса"""
        self.logger.info("Построение индекса OS ...")
        self._ensure_os_index(recreate=self.config.recreate_index)
        self._os_optimize_for_bulk(on=True)
        self._os_bulk_index(data=data, chunk_size=self.config.bulk_chunk_size)
        self._os_optimize_for_bulk(on=False)
        self.logger.info("Индекс OS построен")

    def search(self, body: dict, size: int) -> list[dict]:
        """Поиск при помощи opensearch"""
        resp = self.client.search(index=self.config.index_name, body=body, size=size) # можно заменить на self.os_schema.index_name - но только если recreate будет стоять всегда, что не оч хорошо
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

    def _os_bulk_index(self, data: list[dict[str, tp.Any]], chunk_size: int = 1000) -> None:
        chunk_size = chunk_size or self.os_schema.bulk_chunk_size
        idx_name = self.os_schema.index_name
        id_field = self.os_schema.id_field

        # Быстрый доступ к типам из мэппинга
        props = (self.os_schema.mappings.get("properties") or {})
        type_of: dict[str, str] = {k: (v.get("type") or "") for k, v in props.items()}

        def coerce(field: str, value: tp.Any) -> tp.Any:
            t = type_of.get(field, "")
            if value is None:
                return None
            try:
                if t in ("keyword", "text"):
                    return str(value)
                if t in ("integer", "short", "byte", "long"):
                    return int(value)
                if t in ("float", "half_float", "scaled_float", "double"):
                    return float(value)
                if t == "boolean":
                    return bool(value)
                return value
            except Exception:
                return None

        def gen_actions() -> tp.Iterator[dict[str, tp.Any]]:
            for row in data:
                # Создаем документ, конвертируя значения в соответствии с типами
                doc = {field: coerce(field, value) for field, value in row.items() if field in type_of}

                _id = doc.get(id_field)
                yield {
                    "_op_type": "index",
                    "_index": idx_name,
                    **({"_id": _id} if _id is not None else {}),
                    "_source": doc,
                }

        os_helpers.bulk(self.client, gen_actions(), chunk_size=chunk_size, request_timeout=120)

    def upsert(self, data: list[dict[str, tp.Any]]) -> None:
        """Upsert документов в OpenSearch (обновляет или добавляет)"""
        if not data:
            self.logger.info("Нет данных для upsert в OS")
            return

        self.logger.info(f"🔄 Upsert {len(data)} документов в OS ...")

        # Убедимся, что индекс существует
        self._ensure_os_index(recreate=False)

        # Оптимизация перед bulk-записью
        self._os_optimize_for_bulk(on=True)
        self._os_bulk_upsert(data=data, chunk_size=self.config.bulk_chunk_size)
        self._os_optimize_for_bulk(on=False)

        self.logger.info("✅ Upsert OS завершен")

    def _os_bulk_upsert(self, data: list[dict[str, tp.Any]], chunk_size: int = 1000) -> None:
        idx_name = self.os_schema.index_name
        id_field = self.os_schema.id_field
        props = self.os_schema.mappings.get("properties", {})
        type_of: dict[str, str] = {k: (v.get("type") or "") for k, v in props.items()}

        def coerce(field: str, value: tp.Any) -> tp.Any:
            t = type_of.get(field, "")
            if value is None:
                return None
            try:
                if t in ("keyword", "text"):
                    return str(value)
                if t in ("integer", "short", "byte", "long"):
                    return int(value)
                if t in ("float", "half_float", "scaled_float", "double"):
                    return float(value)
                if t == "boolean":
                    return bool(value)
                return value
            except Exception:
                return None

        def gen_actions():
            for row in data:
                doc = {field: coerce(field, value) for field, value in row.items() if field in type_of}
                _id = doc.get(id_field)
                if _id is None:
                    continue
                yield {
                    "_op_type": "update",
                    "_index": idx_name,
                    "_id": _id,
                    "doc": doc,
                    "doc_as_upsert": True,
                }

        os_helpers.bulk(self.client, gen_actions(), chunk_size=chunk_size, request_timeout=120)

    def _ensure_os_index(self, recreate: bool = False) -> None:
        """Создание индекса с русским анализатором.
        OS_INDEX_ANSWER=true|false — индексировать ли поле answer как text (иначе останется в _source).
        """
        idx_name = self.os_schema.index_name

        if recreate and self.client.indices.exists(index=idx_name, expand_wildcards="all"):
            self.client.indices.delete(index=idx_name, ignore=[400, 404])

        if not self.client.indices.exists(index=idx_name):
            body = {
                "settings": self.os_schema.settings,
                "mappings": self.os_schema.mappings,
            }
            self.client.indices.create(index=idx_name, body=body)

    def _load_os_schema_from_json(self, path: str) -> OSIndexSchema:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return OSIndexSchema(
            index_name=data.get("index_name", self.config.index_name),
            id_field=data.get("id_field", "row_idx"),
            settings=data.get("settings", {}),
            mappings=data.get("mappings", {}),
            bulk_chunk_size=(data.get("bulk") or {}).get("chunk_size", self.config.bulk_chunk_size),
        )

    def fetch_existing(self, size: int = 10000) -> list[dict[str, tp.Any]]:
        """Получить все документы из индекса OpenSearch"""
        idx_name = self.os_schema.index_name
        self.logger.info(f"📥 Чтение документов из индекса {idx_name} ...")
        try:
            query = {"query": {"match_all": {}}}
            resp = self.client.search(index=idx_name, body=query, size=size)
            hits = resp.get("hits", {}).get("hits", [])
            docs = [
                {
                    "ext_id": h["_id"],
                    **h["_source"],
                }
                for h in hits
            ]
            self.logger.info(f"📄 Получено {len(docs)} документов из OS")
            return docs
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка при чтении OS индекса {idx_name}: {e}")
            return []

    def delete(self, ext_ids: list[str]) -> None:
        """
        Удаляет документы из OpenSearch по полю ext_id.
        """
        idx_name = self.os_schema.index_name

        if not ext_ids:
            raise ValueError("Нужно передать ext_ids для удаления")

        self.logger.info(f"🧹 Удаление документов из {idx_name} по ext_id ...")

        try:
            # Оптимизация на время bulk-операций
            self._os_optimize_for_bulk(on=True)

            # Формируем запрос delete_by_query
            query = {
                "terms": {
                    "ext_id": ext_ids
                }
            }

            resp = self.client.delete_by_query(
                index=idx_name,
                body={"query": query},
                refresh=True,
                conflicts="proceed",  # не падает на конфликтных документах
                request_timeout=300,
            )
            deleted_count = resp.get("deleted", 0)
            self.logger.info(f"🗑 Удалено документов по ext_id: {deleted_count}")

        except Exception as e:
            self.logger.error(f"Ошибка при удалении из OS по ext_id: {e}")

        finally:
            self._os_optimize_for_bulk(on=False)