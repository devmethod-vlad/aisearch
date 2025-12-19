import json
import time
import traceback
import typing as tp

from opensearchpy import (
    OpenSearch,
    helpers as os_helpers,
)

from app.common.logger import AISearchLogger
from app.domain.schemas.open_search import OSIndexSchema
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
            http_auth=(
                (self.config.user, self.config.password) if self.config.user else None
            ),
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
        resp = self.client.search(
            index=self.config.index_name, body=body, size=size
        )  # можно заменить на self.os_schema.index_name - но только если recreate будет стоять всегда, что не оч хорошо
        return resp["hits"]["hits"]

    def _os_optimize_for_bulk(self, on: bool) -> None:
        if on:
            self.client.indices.put_settings(
                index=self.config.index_name,
                body={
                    "index": {"refresh_interval": "-1", "translog.durability": "async"}
                },
            )
        else:
            self.client.indices.put_settings(
                index=self.config.index_name,
                body={
                    "index": {
                        "refresh_interval": "1s",
                        "translog.durability": "request",
                    }
                },
            )
            self.client.indices.refresh(index=self.config.index_name)

    def _os_bulk_index(
        self, data: list[dict[str, tp.Any]], chunk_size: int = 1000
    ) -> None:
        chunk_size = chunk_size or self.os_schema.bulk_chunk_size
        idx_name = self.os_schema.index_name
        id_field = self.os_schema.id_field

        # Быстрый доступ к типам из мэппинга
        props = self.os_schema.mappings.get("properties") or {}
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
                doc = {
                    field: coerce(field, value)
                    for field, value in row.items()
                    if field in type_of
                }

                # Формируем _id на основе id_field
                _id = None
                if isinstance(id_field, list):
                    # Составной ID: берем значения полей и объединяем через "::"
                    id_parts = []
                    for field_name in id_field:
                        value = doc.get(field_name)
                        if value is not None:
                            id_parts.append(str(value))
                    if id_parts:
                        _id = "::".join(id_parts)
                else:
                    # Простой ID: берем значение одного поля
                    _id = doc.get(id_field)

                yield {
                    "_op_type": "index",
                    "_index": idx_name,
                    **({"_id": _id} if _id is not None else {}),
                    "_source": doc,
                }

        os_helpers.bulk(
            self.client, gen_actions(), chunk_size=chunk_size, request_timeout=120
        )

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

    def _os_bulk_upsert(
        self, data: list[dict[str, tp.Any]], chunk_size: int = 1000
    ) -> None:
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

        def gen_actions() -> tp.Iterator[dict[str, tp.Any]]:
            for row in data:
                doc = {
                    field: coerce(field, value)
                    for field, value in row.items()
                    if field in type_of
                }

                # Формируем _id на основе id_field
                _id = None
                if isinstance(id_field, list):
                    # Составной ID
                    id_parts = []
                    for field_name in id_field:
                        value = doc.get(field_name)
                        if value is not None:
                            id_parts.append(str(value))
                    if id_parts:
                        _id = "::".join(id_parts)
                else:
                    # Простой ID
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

        os_helpers.bulk(
            self.client, gen_actions(), chunk_size=chunk_size, request_timeout=120
        )

    def _ensure_os_index(self, recreate: bool = False) -> None:
        """Создание индекса с русским анализатором.
        OS_INDEX_ANSWER=true|false — индексировать ли поле answer как text (иначе останется в _source).
        """
        idx_name = self.os_schema.index_name

        if recreate and self.client.indices.exists(
            index=idx_name, expand_wildcards="all"
        ):
            self.client.indices.delete(index=idx_name, ignore=[400, 404])

        if not self.client.indices.exists(index=idx_name):
            body = {
                "settings": self.os_schema.settings,
                "mappings": self.os_schema.mappings,
            }
            self.client.indices.create(index=idx_name, body=body)

    def _load_os_schema_from_json(self, path: str) -> OSIndexSchema:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Получаем id_field - может быть строкой или списком
        id_field = data.get("id_field", "row_idx")

        # Если id_field задан как строка "source,ext_id", преобразуем в список
        if isinstance(id_field, str) and "," in id_field:
            id_field = [f.strip() for f in id_field.split(",")]

        return OSIndexSchema(
            index_name=data.get("index_name", self.config.index_name),
            id_field=id_field,  # Может быть строкой или списком
            settings=data.get("settings", {}),
            mappings=data.get("mappings", {}),
            bulk_chunk_size=(data.get("bulk") or {}).get(
                "chunk_size", self.config.bulk_chunk_size
            ),
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
                    "ext_id": h["_id"].split("::")[1] if "::" in h["_id"] else h["_id"],
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
        """Удаляет документы из OpenSearch по полю ext_id."""
        idx_name = self.os_schema.index_name

        if not ext_ids:
            raise ValueError("Нужно передать ext_ids для удаления")

        self.logger.info(f"🧹 Удаление документов из {idx_name} по ext_id ...")

        try:
            # Оптимизация на время bulk-операций
            self._os_optimize_for_bulk(on=True)

            # Формируем запрос delete_by_query
            query = {"terms": {"ext_id": ext_ids}}

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
            self.logger.error(
                f"❌ Ошибка при удалении из OS по ext_id ({type(e)}): {traceback.format_exc()}"
            )

        finally:
            self._os_optimize_for_bulk(on=False)

    def ensure_index_not_empty(self) -> None:
        """Проверяет, что индекс существует и содержит хотя бы 1 документ."""
        idx = self.os_schema.index_name

        try:
            # Обновляем индекс, чтобы bulk стал видимым
            self.client.indices.refresh(index=idx)

            # Быстрый подсчёт документов
            resp = self.client.count(index=idx)
            count = resp.get("count", 0)

            if count == 0:
                self.logger.error(f"❌ Индекс {idx} пуст после индексации!")
                raise RuntimeError(f"Индекс {idx} пуст после индексации")

            self.logger.info(f"📦 Индекс {idx} содержит {count} документов")

        except Exception as e:
            raise RuntimeError(f"Ошибка при проверке содержимого индекса {idx}: {e}")

    def _resolve_keyword_field(self, idx_name: str, field: str) -> str | None:
        """Возвращает поле для exact-совпадений (terms): либо сам field, если type=keyword,
        либо field+'.keyword', если type=text и есть подполе keyword,
        иначе None (будем падать в режим полного scan).
        """
        try:
            mapping = self.client.indices.get_mapping(index=idx_name)
            # Если idx_name — алиас, берём первое реальное имя индекса
            m = next(iter(mapping.values()))  # {'mappings': {...}}
            props = (m.get("mappings") or {}).get("properties") or {}

            # Если поле содержит точку (например, "source.keyword")
            if "." in field:
                parts = field.split(".")
                current = props
                for part in parts[:-1]:
                    if part in current:
                        current = current[part].get("properties") or current[part]
                    else:
                        return None
                finfo = current.get(parts[-1], {})
            else:
                finfo = props.get(field, {})

            if finfo.get("type") == "keyword":
                return field

            # text с подполем keyword?
            sub = finfo.get("fields") or {}
            if "keyword" in sub and (sub["keyword"].get("type") == "keyword"):
                return f"{field}.keyword"

        except Exception as e:
            self.logger.warning(f"Не удалось получить mapping для поля '{field}': {e}")

        return None

    def ids_exist_by_source_field(
        self,
        incoming_ext_ids: tp.Iterable[tp.Any],
        source: str = None,
        field: str = "ext_id",
        batch_size: int = 2000,
        scan_page: int = 1000,
        scroll_keepalive: str = "5m",
    ) -> tuple[list[str], list[str], list[str]]:
        """Вернёт три списка (строки):
        - found_incoming: входящие ext_id, найденные в индексе по _source[field]
        - missing_incoming: входящие ext_id, которых нет в индексе
        - extra_in_store: ext_id, которые есть в индексе, но их нет во входящих
        """
        idx_name = self.os_schema.index_name

        # входящие -> строки
        incoming_ids = [str(x) for x in incoming_ext_ids if x is not None]
        incoming_set = set(incoming_ids)

        # Готовим единый term-фильтр по source (если он задан)
        source_filter: dict | None = None
        if source:
            source_term_field = self._resolve_keyword_field(idx_name, "source")
            source_field_for_term = source_term_field or "source"
            source_filter = {"term": {source_field_for_term: source}}

        term_field = self._resolve_keyword_field(idx_name, field)

        # для переиспользования store_ids при сканировании
        store_ids: set[str] | None = None

        # 1) находим «что из входящих есть» с фильтром по source если указан
        found_incoming: set[str] = set()

        if term_field:
            # быстрый путь: точные terms-запросы по keyword/подполю
            base_filters = []
            if source_filter:
                base_filters.append(source_filter)

            for i in range(0, len(incoming_ids), batch_size):
                batch = incoming_ids[i : i + batch_size]

                must_clauses = [{"terms": {term_field: batch}}] + base_filters

                body = {
                    "_source": [field],
                    "query": {"bool": {"filter": must_clauses}},  # filter вместо must
                    "size": len(batch),
                }
                resp = self.client.search(index=idx_name, body=body)

                for hit in resp.get("hits", {}).get("hits", []):
                    src = hit.get("_source") or {}
                    val = src.get(field)
                    if val is not None:
                        found_incoming.add(str(val))
        else:
            # нет точного поля — деградируем: читаем значения поля и пересекаем
            self.logger.warning(
                f"Поле '{field}' не keyword и без подполей keyword — используем scan для пересечения."
            )
            store_ids_scan: set[str] = set()

            scan_query = (
                {"bool": {"filter": [source_filter]}}
                if source_filter
                else {"match_all": {}}
            )
            scan_body = {"_source": [field], "query": scan_query}

            for hit in os_helpers.scan(
                self.client,
                index=idx_name,
                query=scan_body,
                size=scan_page,
                scroll=scroll_keepalive,
            ):
                src = hit.get("_source") or {}
                v = src.get(field)
                if v is not None:
                    store_ids_scan.add(str(v))

            found_incoming = incoming_set & store_ids_scan
            store_ids = store_ids_scan  # запоминаем, чтобы не сканировать второй раз

        missing_incoming = incoming_set - found_incoming

        # 2) extra_in_store — всё, что в индексе с указанным source, но не пришло
        if store_ids is None:
            # ещё не сканировали индекс
            store_ids = set()

            scan_query = (
                {"bool": {"filter": [source_filter]}}
                if source_filter
                else {"match_all": {}}
            )
            scan_body = {"_source": [field], "query": scan_query}

            for hit in os_helpers.scan(
                self.client,
                index=idx_name,
                query=scan_body,
                size=scan_page,
                scroll=scroll_keepalive,
            ):
                src = hit.get("_source") or {}
                v = src.get(field)
                if v is not None:
                    store_ids.add(str(v))

        extra_in_store = store_ids - incoming_set

        return (
            list(found_incoming),
            list(missing_incoming),
            list(extra_in_store),
        )

    def delete_by_ext_ids(
        self,
        ext_ids: list[str],
        field: str = "ext_id",
        batch_size: int = 2000,
        scan_page: int = 1000,
        scroll_keepalive: str = "5m",
    ) -> int:
        """Удаляет документы по строковому полю _source[field].
        Возвращает количество удалённых документов.
        """
        idx_name = self.os_schema.index_name
        if not ext_ids:
            return 0

        # приводим вход к строкам
        wanted = [str(x) for x in ext_ids if x is not None]
        wanted_set = set(wanted)

        deleted_total = 0
        term_field = self._resolve_keyword_field(idx_name, field)

        # На bulk-операции ускоряем индекс
        self._os_optimize_for_bulk(on=True)
        try:
            if term_field:
                # Быстрый и точный путь: delete_by_query terms батчами
                for i in range(0, len(wanted), batch_size):
                    batch = wanted[i : i + batch_size]

                    # Формируем фильтр
                    terms_dict: dict[str, list[str]] = {term_field: batch}

                    resp = self.client.delete_by_query(
                        index=idx_name,
                        body={"query": {"terms": terms_dict}},
                        refresh=False,  # НЕ обновляем после каждого батча
                        conflicts="proceed",
                        request_timeout=300,
                    )
                    deleted_total += int(resp.get("deleted", 0))
            else:
                # Фолбэк: найдём _id документов по scan и удалим по _id
                ids_to_delete: set[str] = set()

                scan_body = {"_source": [field], "query": {"match_all": {}}}

                for hit in os_helpers.scan(
                    self.client,
                    index=idx_name,
                    query=scan_body,
                    size=scan_page,
                    scroll=scroll_keepalive,
                ):
                    src = hit.get("_source") or {}
                    v = src.get(field)
                    if v is not None and str(v) in wanted_set:
                        ids_to_delete.add(hit["_id"])

                if ids_to_delete:

                    def gen_actions() -> tp.Generator[dict[str, str | int], None, None]:
                        for _id in ids_to_delete:
                            yield {"_op_type": "delete", "_index": idx_name, "_id": _id}

                    success, _ = os_helpers.bulk(
                        self.client,
                        gen_actions(),
                        chunk_size=batch_size,
                        request_timeout=300,
                    )
                    deleted_total += int(success)
        except Exception as e:
            self.logger.error(
                f"❌ Ошибка при удалении из OS по ext_id ({type(e)}): {traceback.format_exc()}"
            )
        finally:
            # Делаем refresh один раз в конце
            try:
                self.client.indices.refresh(index=idx_name)
            except Exception as e:
                self.logger.warning(f"Ошибка при refresh индекса: {e}")
            self._os_optimize_for_bulk(on=False)

        self.logger.info(f"🗑 OS: удалено {deleted_total} документов по {field}")
        return deleted_total

    def diff_modified_by_ext_ids(
        self,
        incoming_modified: dict[str, str],
        *,
        field: str = "ext_id",
        modified_field: str = "modified_at",
        batch_size: int = 2000,
        scan_page: int = 1000,
        scroll_keepalive: str = "5m",
    ) -> list[str]:
        """Вернёт список ext_id, у которых modified_at в индексе OpenSearch отличается
        от входящего значения.
        """
        idx_name = self.os_schema.index_name

        # входящие -> строки
        incoming_map = {
            str(k): ("" if v is None else str(v)) for k, v in incoming_modified.items()
        }
        wanted_ids = list(incoming_map.keys())
        incoming_set = set(wanted_ids)

        diffs: set[str] = set()
        term_field = self._resolve_keyword_field(idx_name, field)

        if term_field:
            # быстрый точный путь
            for i in range(0, len(wanted_ids), batch_size):
                batch = wanted_ids[i : i + batch_size]
                if not batch:
                    continue

                # Простой terms запрос без фильтра по source
                terms_dict: dict[str, list[str]] = {term_field: batch}
                body = {
                    "_source": [field, modified_field],
                    "query": {"terms": terms_dict},
                    "size": len(batch),
                }
                resp = self.client.search(index=idx_name, body=body)
                for hit in resp.get("hits", {}).get("hits", []):
                    src = hit.get("_source") or {}
                    ext_val = src.get(field)
                    if ext_val is None:
                        continue
                    ext = str(ext_val)
                    idx_mod = (
                        ""
                        if src.get(modified_field) is None
                        else str(src.get(modified_field))
                    )
                    inc_mod = incoming_map.get(ext)
                    if inc_mod is None:
                        continue
                    if idx_mod != inc_mod:
                        diffs.add(ext)
        else:
            # фолбэк: сканируем индекс, берём только нужные поля и сравниваем на лету
            scan_body = {"_source": [field, modified_field], "query": {"match_all": {}}}
            for hit in os_helpers.scan(
                self.client,
                index=idx_name,
                query=scan_body,
                size=scan_page,
                scroll=scroll_keepalive,
            ):
                src = hit.get("_source") or {}
                ext_val = src.get(field)
                if ext_val is None:
                    continue
                ext = str(ext_val)
                if ext not in incoming_set:
                    continue
                idx_mod = (
                    ""
                    if src.get(modified_field) is None
                    else str(src.get(modified_field))
                )
                inc_mod = incoming_map[ext]
                if idx_mod != inc_mod:
                    diffs.add(ext)

        return list(diffs)
