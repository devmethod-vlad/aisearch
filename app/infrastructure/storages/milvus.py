import contextlib
import time
import traceback
import typing as tp

import numpy as np
from pymilvus import (
    AsyncMilvusClient,
    Collection,
    CollectionSchema,
    DataType,
)
from pymilvus.grpc_gen.common_pb2 import ConsistencyLevel
from pymilvus.milvus_client import IndexParams
from sentence_transformers import SentenceTransformer

from app.common.logger import AISearchLogger
from app.infrastructure.storages.interfaces import IVectorDatabase
from app.infrastructure.utils.metrics import metrics_print
from app.infrastructure.utils.milvus import load_schema_and_indexes_from_json
from app.infrastructure.utils.nlp import l2_normalize
from app.settings.config import MilvusSettings


class MilvusDatabase(IVectorDatabase):
    def __init__(self, settings: MilvusSettings, logger: AISearchLogger):
        milvus_init_start = time.perf_counter()
        self.config = settings
        self.logger = logger
        self.client = AsyncMilvusClient(
            uri=f"http{'s' if self.config.use_ssl else ''}://{self.config.host}:{self.config.port}",
            timeout=self.config.connection_timeout,
        )
        self.__collections_loaded = set()
        self._search_params_by_field = {}
        metrics_print("🕒 Инициализация Milvus", milvus_init_start)

    def _with_timeout(self, kwargs: dict[str, tp.Any]) -> dict[str, tp.Any]:
        """Добавляет timeout по умолчанию, если он не указан."""
        if kwargs.get("timeout") is None:
            kwargs = kwargs | {"timeout": self.config.query_timeout}
        return kwargs

    async def _load_collection(
        self, collection_name: str, **kwargs: dict[str, tp.Any]
    ) -> None:
        return await self.client.load_collection(
            collection_name,
            consistency_level=ConsistencyLevel.Strong,
            **self._with_timeout(kwargs),
        )

    async def _has_collection(
        self, collection_name: str, **kwargs: dict[str, tp.Any]
    ) -> bool:
        return await self.client.has_collection(
            collection_name,
            **self._with_timeout(kwargs),
        )

    async def _create_collection(
        self,
        collection_name: str,
        schema: CollectionSchema,
        **kwargs: dict[str, tp.Any],
    ) -> None:
        return await self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            consistency_level=ConsistencyLevel.Strong,
            **self._with_timeout(kwargs),
        )

    async def _insert(
        self, collection_name: str, data: list[dict], **kwargs: dict[str, tp.Any]
    ) -> dict:
        return await self.client.insert(
            collection_name=collection_name,
            data=data,
            **self._with_timeout(kwargs),
        )

    async def _upsert(
        self, collection_name: str, data: list[dict], **kwargs: dict[str, tp.Any]
    ) -> dict:
        return await self.client.upsert(
            collection_name=collection_name,
            data=data,
            **self._with_timeout(kwargs),
        )

    async def _search(self, collection_name: str, **kwargs: dict[str, tp.Any]) -> list:
        return await self.client.search(
            collection_name=collection_name,
            consistency_level=ConsistencyLevel.Bounded,
            **self._with_timeout(kwargs),
        )

    async def _query(self, collection_name: str, **kwargs: dict[str, tp.Any]) -> list:
        return await self.client.query(
            collection_name=collection_name,
            consistency_level=ConsistencyLevel.Strong,
            **self._with_timeout(kwargs),
        )

    async def _delete(self, collection_name: str, **kwargs: dict[str, tp.Any]) -> dict:
        return await self.client.delete(
            collection_name=collection_name,
            consistency_level=ConsistencyLevel.Strong,
            **self._with_timeout(kwargs),
        )

    async def _get(
        self, collection_name: str, ids: list, **kwargs: dict[str, tp.Any]
    ) -> list[dict]:
        return await self.client.get(
            collection_name=collection_name,
            ids=ids,
            consistency_level=ConsistencyLevel.Strong,
            **self._with_timeout(kwargs),
        )

    async def _flush(self, collection_name: str, **kwargs: dict[str, tp.Any]) -> None:
        return await self.client.flush(
            collection_name,
            **self._with_timeout(kwargs),
        )

    async def _create_index(
        self,
        collection_name: str,
        index_params: IndexParams,
        **kwargs: dict[str, tp.Any],
    ) -> None:
        return await self.client.create_index(
            collection_name=collection_name,
            index_params=index_params,
            **self._with_timeout(kwargs),
        )

    async def _list_collections(self, **kwargs: dict[str, tp.Any]) -> list[str]:
        return await self.client.list_collections(
            **self._with_timeout(kwargs),
        )

    async def _list_indexes(
        self, collection_name: str, **kwargs: dict[str, tp.Any]
    ) -> list:
        return await self.client.list_indexes(
            collection_name,
            **self._with_timeout(kwargs),
        )

    async def _describe_collection(
        self, collection_name: str, **kwargs: dict[str, tp.Any]
    ) -> dict:
        return await self.client.describe_collection(
            collection_name,
            **self._with_timeout(kwargs),
        )

    async def _get_collection_stats(
        self, collection_name: str, **kwargs: dict[str, tp.Any]
    ) -> dict:
        return await self.client.get_collection_stats(
            collection_name,
            **self._with_timeout(kwargs),
        )

    async def _drop_collection(
        self, collection_name: str, **kwargs: dict[str, tp.Any]
    ) -> None:
        return await self.client.drop_collection(
            collection_name,
            **self._with_timeout(kwargs),
        )

    async def _release_collection(
        self, collection_name: str, **kwargs: dict[str, tp.Any]
    ) -> None:
        return await self.client.release_collection(
            collection_name,
            **self._with_timeout(kwargs),
        )

    @staticmethod
    def _prepare_index_params(
        field_name: str = "", **kwargs: dict[str, tp.Any]
    ) -> IndexParams:
        return AsyncMilvusClient.prepare_index_params(field_name, **kwargs)

    @staticmethod
    def get_model_name(model: SentenceTransformer) -> str:
        return model._first_module().auto_model.config._name_or_path.split("/")[-1]

    async def load_collection(self, collection_name: str) -> None:
        await self._load_collection(collection_name)
        self.__collections_loaded.add(collection_name)

    async def create_collection(self, collection_name: str) -> None:
        if await self._has_collection(collection_name):
            self.logger.info(
                f"Коллекция {collection_name} уже существует, пересоздаем..."
            )
            await self._drop_collection(collection_name)

        fields, index_specs, search_params_by_field = load_schema_and_indexes_from_json(
            self.config.schema_path
        )

        schema = CollectionSchema(fields, description=f"Collection {collection_name}")
        await self._create_collection(
            collection_name=collection_name,
            schema=schema,
        )

        if index_specs:
            field_names = {f.name for f in fields}
            index_params = self._prepare_index_params()

            for idx in index_specs:
                if idx.field_name not in field_names:
                    raise ValueError(
                        f"В JSON указан индекс для отсутствующего поля: {idx.field_name}"
                    )

                add_kwargs = {
                    "field_name": idx.field_name,
                    "index_type": idx.index_type,
                    "params": idx.params or {},
                }
                if idx.metric_type is not None:
                    add_kwargs["metric_type"] = idx.metric_type

                index_params.add_index(**add_kwargs)

            await self._create_index(
                collection_name=collection_name,
                index_params=index_params,
            )
        else:
            self.logger.warning(
                "🚨 Milvus: в JSON не задано ни одного индекса — коллекция будет работать в режиме FLAT (медленнее)."
            )

        self._search_params_by_field = search_params_by_field or {}
        await self.load_collection(collection_name)

    async def insert_vectors(
        self,
        collection_name: str,
        vectors: list[list[float]],
        metadata: list[dict[str, tp.Any]] | None = None,
        batch_size: int = 512,
    ) -> None:
        vectors_size = len(vectors)

        if metadata is not None:
            data_size = len(metadata)
            if vectors_size != data_size:
                raise ValueError(
                    f"Количество векторов не равно количеству метаданных ({vectors_size} != {data_size})"
                )
        else:
            if vectors_size == 0:
                raise ValueError("Нельзя вставить 0 векторов без метаданных")
            metadata = [{} for _ in range(vectors_size)]

        fields, index_specs, search_params_by_field = load_schema_and_indexes_from_json(
            self.config.schema_path
        )
        f_by_name = {f.name: f for f in fields}

        vec_field = self.config.vector_field
        if vec_field not in f_by_name:
            raise ValueError(f"В схеме нет векторного поля '{vec_field}'")

        vfs = f_by_name[vec_field]
        dim = getattr(vfs, "dim", None)
        if dim is None:
            params = getattr(vfs, "params", {}) or {}
            dim = params.get("dim")

        def _normalize_vector(vec: tp.Sequence[tp.Any]) -> list[float]:
            if dim is not None and len(vec) != dim:
                raise ValueError(f"Ожидалась размерность {len(vec)}, получили {dim}")
            try:
                return [float(x) for x in vec]
            except Exception as e:
                raise TypeError(
                    f"Невозможно привести элементы вектора к float: {e}"
                ) from e

        def _coerce(name: str, value: tp.Any) -> tuple[tp.Any | None, bool]:
            f = f_by_name.get(name)
            if f is None:
                return None, True
            if value is None:
                return None, True

            dt = f.dtype
            if dt == DataType.VARCHAR:
                s = str(value)
                max_len = getattr(f, "max_length", None)
                if max_len:
                    s = s[:max_len]
                return s, False
            if dt in (DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64):
                return int(value), False
            if dt in (DataType.FLOAT, DataType.DOUBLE):
                return float(value), False
            if dt == DataType.BOOL:
                return bool(value), False
            if dt == DataType.FLOAT_VECTOR:
                return None, True
            return value, False

        num_batches = (vectors_size + batch_size - 1) // batch_size
        self.logger.info(
            f"Загрузка векторов{' и метаданных' if metadata else ''}, батч {batch_size}, всего батчей: {num_batches}"
        )

        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, vectors_size)

            vectors_batch = vectors[start:end]
            rows_batch = metadata[start:end]

            data: list[dict[str, tp.Any]] = []
            for vec, row in zip(vectors_batch, rows_batch, strict=True):
                item: dict[str, tp.Any] = {}
                item[vec_field] = _normalize_vector(vec)

                for k, v in row.items():
                    val, drop = _coerce(k, v)
                    if not drop:
                        item[k] = val

                data.append(item)

            await self._insert(
                collection_name=collection_name,
                data=data,
            )
            self.logger.info(f"Загружено {i + 1}/{num_batches} батчей")

        await self._flush(collection_name)

    async def search(
        self, collection_name: str, query_vector: list[float], top_k: int
    ) -> list[dict[str, tp.Any]]:
        top_k = max(top_k, 1)

        if collection_name not in self.__collections_loaded:
            await self.load_collection(collection_name)

        results = await self._search(
            collection_name=collection_name,
            data=[query_vector],
            anns_field=self.config.vector_field,
            params={"metric_type": self.config.metric_type, "params": {"ef": 64}},
            limit=top_k,
            output_fields=self.config.output_fields,
        )

        out: list[dict[str, tp.Any]] = []
        if not results:
            return out

        hits = results[0]
        for h in hits:
            fields = h.entity
            row = {k: fields.get(k, "") for k in self.config.output_fields}
            row["score_dense"] = float(h.distance)
            out.append(row)
        return out

    async def collection_ready(self, collection_name: str) -> bool:
        try:
            if not await self._has_collection(collection_name):
                return False

            indexes = await self._list_indexes(collection_name)
            return len(indexes) > 0
        except Exception:
            return False

    async def delete_collection(self, collection_name: str) -> None:
        collections = await self._list_collections(timeout=self.config.query_timeout)
        if collection_name not in collections:
            self.logger.warning(
                f"Коллекция {collection_name} не существует, удаление не требуется."
            )
            self.__collections_loaded.discard(collection_name)
            return

        await self._load_collection(collection_name, timeout=self.config.query_timeout)
        await self._drop_collection(collection_name, timeout=self.config.query_timeout)
        self.__collections_loaded.discard(collection_name)

    async def preload_collections(self) -> None:
        collection_name = self.config.collection_name
        try:
            self.logger.info(f"⏳ Загрузка коллекции {collection_name} ...")
            await self.load_collection(collection_name)
            self.logger.info(f"✅ Коллекция {collection_name} успешно загружена")
        except Exception as e:
            self.logger.warning(
                f"⚠️ Не удалось загрузить коллекцию {collection_name}: {e}"
            )

    async def index_documents(
        self,
        collection_name: str,
        model: SentenceTransformer,
        documents: list[str],
        metadata: list[dict[str, tp.Any]] | None = None,
    ) -> None:
        self.logger.info("Происходит индексация документов в Milvus ...")

        embeddings = await self.get_embeddings(model, documents)
        await self.create_collection(collection_name)

        await self.insert_vectors(
            collection_name=collection_name,
            vectors=embeddings.tolist(),
            metadata=metadata,
        )

    async def get_embeddings(
        self, model: SentenceTransformer, documents: list[str]
    ) -> np.ndarray:
        """Получение эмбеддингов"""
        embeddings = model.encode(documents, normalize_embeddings=True)
        embeddings = np.vstack([l2_normalize(e) for e in embeddings])
        return embeddings

    async def initialize_collection(
        self,
        collection_name: str,
        model: SentenceTransformer,
        documents: list[str],
        metadata: list[dict[str, tp.Any]] | None = None,
    ) -> None:
        self.logger.info(f"Инициализация коллекции {collection_name} ...")
        await self.index_documents(
            collection_name=collection_name,
            model=model,
            documents=documents,
            metadata=metadata,
        )
        await self._release_collection(collection_name)
        await self.load_collection(collection_name)

    async def close(self) -> None:
        await self.client.close()

    async def fetch_existing(
        self, collection_name: str, output_fields: list[str] | None = None
    ) -> list[dict]:
        try:
            if collection_name not in self.__collections_loaded:
                await self.load_collection(collection_name)

            output_fields = (
                output_fields or self.config.output_fields.split(",")
                if isinstance(self.config.output_fields, str)
                else self.config.output_fields
            )

            results = []
            batch_size = 8_192
            last_pk = -1

            while True:
                filter_expr = f"pk > {last_pk}"
                batch_res = await self._query(
                    collection_name=collection_name,
                    filter=filter_expr,
                    output_fields=output_fields,
                    limit=batch_size,
                )
                if not batch_res:
                    break
                results.extend(batch_res)
                last_pk = max(r["pk"] for r in batch_res)

            self.logger.info(f"Получено {len(results)} записей из Milvus")
            return results

        except Exception as e:
            self.logger.error(
                f"Не удалось получить записи мильвус {collection_name} ({type(e)}): {traceback.format_exc()}"
            )
            return []

    async def upsert_vectors(
        self,
        collection_name: str,
        vectors: list[list[float]],
        metadata: list[dict[str, tp.Any]] | None = None,
        batch_size: int = 512,
    ) -> None:
        if not metadata:
            metadata = [{} for _ in vectors]
        elif len(metadata) != len(vectors):
            raise ValueError(
                "Количество векторов не совпадает с количеством метаданных"
            )

        existing = await self.fetch_existing(
            collection_name, output_fields=["ext_id", "pk"]
        )
        ext_id_to_pk = {r["ext_id"]: r["pk"] for r in existing}

        pk_to_delete = [
            ext_id_to_pk[m["ext_id"]] for m in metadata if m["ext_id"] in ext_id_to_pk
        ]
        if pk_to_delete:
            filter_expr = "pk in [" + ",".join(map(str, pk_to_delete)) + "]"
            await self._delete(
                collection_name=collection_name,
                filter=filter_expr,
            )
            await self._flush(collection_name)

        fields, _, _ = load_schema_and_indexes_from_json(self.config.schema_path)
        f_by_name = {f.name: f for f in fields}
        vec_field = self.config.vector_field
        dim = getattr(
            f_by_name[vec_field], "dim", f_by_name[vec_field].params.get("dim")
        )

        def _normalize(vec: list[tp.Any]) -> list[float]:
            if len(vec) != dim:
                raise ValueError(f"Ожидалась размерность {len(vec)}, получили {dim}")
            return [float(x) for x in vec]

        def _coerce(name: str, value: tp.Any) -> tuple[tp.Any | None, bool]:
            f = f_by_name.get(name)
            if f is None or value is None:
                return None, True
            if f.dtype == DataType.VARCHAR:
                s = str(value)
                if getattr(f, "max_length", None):
                    s = s[: f.max_length]
                return s, False
            if f.dtype in (
                DataType.INT64,
                DataType.INT32,
                DataType.INT16,
                DataType.INT8,
            ):
                return int(value), False
            if f.dtype in (DataType.FLOAT, DataType.DOUBLE):
                return float(value), False
            if f.dtype == DataType.BOOL:
                return bool(value), False
            return value, False

        total = len(vectors)
        num_batches = (total + batch_size - 1) // batch_size
        self.logger.info(
            f"Upsert {total} векторов/метаданных, батч {batch_size}, всего {num_batches} батчей"
        )

        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, total)
            vec_batch = [_normalize(v) for v in vectors[start:end]]
            meta_batch = metadata[start:end]

            data = []
            for vec, row in zip(vec_batch, meta_batch, strict=True):
                item: dict[str, tp.Any] = {vec_field: vec}
                for k, v in row.items():
                    val, drop = _coerce(k, v)
                    if not drop:
                        item[k] = val
                data.append(item)

            await self._insert(
                collection_name=collection_name,
                data=data,
            )
            self.logger.info(f"Загружено {i + 1}/{num_batches} батчей")

        await self._flush(collection_name)

    async def delete_vectors(
        self,
        collection_name: str,
        ext_ids: list[str] | None = None,
        filter_expr: str | None = None,
    ) -> None:
        if not ext_ids and not filter_expr:
            raise ValueError("Нужно передать либо ext_ids, либо filter_expr")

        if collection_name not in self.__collections_loaded:
            await self.load_collection(collection_name)

        if ext_ids:
            quoted_ids = ",".join(f"'{x}'" for x in ext_ids)
            filter_expr = f"ext_id in [{quoted_ids}]"

        self.logger.info(
            f"🧹 Удаление записей из {collection_name} по фильтру: {filter_expr}"
        )

        try:
            await self._delete(
                collection_name=collection_name,
                filter=filter_expr,
            )
            await self._flush(collection_name)
            self.logger.info(f"✅ Удаление завершено ({collection_name})")
        except Exception as e:
            self.logger.error(
                f"Ошибка при удалении из {collection_name} ({type(e)}): {traceback.format_exc()}"
            )

    async def collection_not_empty(self, collection_name: str) -> bool:
        try:
            existing = await self.fetch_existing(collection_name, output_fields=["pk"])
            row_count = len(existing)
            self.logger.info(f"Milvus: строк по сегментам = {row_count}")
            return row_count > 0
        except Exception as e:
            self.logger.warning(
                f"Не удалось проверить коллекцию {collection_name}: {e}"
            )
            return False

    def _ensure_varchar_field(self, col: Collection, field: str) -> None:
        for f in col.schema.fields:
            if f.name == field:
                if f.dtype != DataType.VARCHAR:
                    raise TypeError(
                        f"Поле '{field}' в Milvus должно быть строковым (VarChar), а сейчас: {f.dtype}"
                    )
                return
        raise ValueError(f"В коллекции нет поля '{field}'")

    def _escape_str_for_expr(self, s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    async def find_existing_ext_ids(
        self,
        collection_name: str,
        incoming_ext_ids: tp.Iterable[tp.Any],
        field: str = "ext_id",
        batch_size: int = 1000,
        iterator_batch: int = 4096,
        source_field: str | None = None,
        source: str | None = None,
    ) -> tuple[list[str], list[str], list[str]]:
        schema = await self._describe_collection(collection_name)
        fields = {f["name"]: f for f in schema["fields"]}

        if field not in fields:
            raise RuntimeError(
                f"Field '{field}' not found in collection '{collection_name}'"
            )

        if fields[field]["type"] != DataType.VARCHAR:
            raise RuntimeError(
                f"Field '{field}' must be VarChar, got {fields[field]['type']}"
            )

        if (source is None) != (source_field is None):
            raise ValueError(
                "Both 'source' and 'source_field' must be provided together or omitted"
            )

        if source is not None and source_field is not None:
            if source_field not in fields:
                raise RuntimeError(
                    f"Field '{source_field}' not found in collection '{collection_name}'"
                )

            if fields[source_field]["type"] != DataType.VARCHAR:
                raise RuntimeError(
                    f"Field '{source_field}' must be VarChar, got {fields[source_field]['type']}"
                )

        with contextlib.suppress(Exception):
            await self.load_collection(collection_name)

        incoming_ids = [str(x) for x in incoming_ext_ids if x is not None]
        incoming_set = set(incoming_ids)
        found_incoming: set[str] = set()

        escaped_source_value: str | None = None
        if source is not None:
            escaped_source_value = self._escape_str_for_expr(source)

        for i in range(0, len(incoming_ids), batch_size):
            batch = incoming_ids[i : i + batch_size]

            list_literal = ",".join(f'"{self._escape_str_for_expr(s)}"' for s in batch)
            base_expr = f"{field} in [{list_literal}]"

            if escaped_source_value is not None and source_field is not None:
                filter_expr = (
                    f"({base_expr}) AND {source_field} == " f'"{escaped_source_value}"'
                )
            else:
                filter_expr = base_expr

            res = await self._query(
                collection_name=collection_name,
                filter=filter_expr,
                output_fields=[field],
                offset=0,
                limit=batch_size,
            )

            for r in res:
                v = r.get(field)
                if v is not None:
                    found_incoming.add(str(v))

        missing_incoming = incoming_set - found_incoming

        extra_in_store: set[str] = set()
        store_ids: set[str] = set()

        if escaped_source_value is not None and source_field is not None:
            extras_filter_expr = f'{source_field} == "{escaped_source_value}"'
        else:
            extras_filter_expr = ""

        offset = 0
        while True:
            res = await self._query(
                collection_name=collection_name,
                filter=extras_filter_expr,
                output_fields=[field],
                offset=offset,
                limit=iterator_batch,
            )

            if not res:
                break

            for r in res:
                v = r.get(field)
                if v is not None:
                    store_ids.add(str(v))

            offset += iterator_batch

        extra_in_store = store_ids - incoming_set

        return (
            list(found_incoming),
            list(missing_incoming),
            list(extra_in_store),
        )

    async def delete_by_ext_ids(
        self,
        collection_name: str,
        ext_ids: list[str],
        field: str = "ext_id",
        batch_size: int = 1000,
    ) -> int:
        if not ext_ids:
            return 0

        wanted = [str(x) for x in ext_ids if x is not None]
        if not wanted:
            return 0

        deleted_total = 0

        for i in range(0, len(wanted), batch_size):
            batch = wanted[i : i + batch_size]

            list_literal = ",".join(f'"{self._escape_str_for_expr(v)}"' for v in batch)
            filter_expr = f"{field} in [{list_literal}]"

            try:
                mr = await self._delete(
                    collection_name=collection_name,
                    filter=filter_expr,
                )
                deleted_total += int(mr.get("delete_count", 0))

            except Exception as e:
                self.logger.error(
                    f"Milvus delete batch failed in collection "
                    f"'{collection_name}' ({type(e)}): {traceback.format_exc()}"
                )
                continue

        self.logger.info(
            f"🗑 Milvus: удалено ~{deleted_total} entities по полю '{field}'"
        )
        return deleted_total

    async def diff_modified_by_ext_ids(
        self,
        collection_name: str,
        incoming_modified: dict[str, str],
        *,
        field: str = "ext_id",
        modified_field: str = "modified_at",
        batch_size: int = 1000,
    ) -> list[str]:
        schema = await self._describe_collection(collection_name)
        fields = {f["name"]: f for f in schema["fields"]}

        if field not in fields:
            raise RuntimeError(
                f"Field '{field}' not found in collection '{collection_name}'"
            )
        if modified_field not in fields:
            raise RuntimeError(
                f"Field '{modified_field}' not found in collection '{collection_name}'"
            )
        if fields[field]["type"] != DataType.VARCHAR:
            raise RuntimeError(
                f"Field '{field}' must be VarChar, got {fields[field]['type']}"
            )
        if fields[modified_field]["type"] != DataType.VARCHAR:
            raise RuntimeError(f"Field '{modified_field}' must be VarChar")

        with contextlib.suppress(Exception):
            await self.load_collection(collection_name)

        incoming_map = {
            str(k): ("" if v is None else str(v)) for k, v in incoming_modified.items()
        }
        ids = list(incoming_map.keys())

        diffs: set[str] = set()

        for i in range(0, len(ids), batch_size):
            batch = ids[i : i + batch_size]

            list_literal = ",".join(f'"{self._escape_str_for_expr(x)}"' for x in batch)
            filter_expr = f"{field} in [{list_literal}]"

            rows = await self._query(
                collection_name=collection_name,
                filter=filter_expr,
                output_fields=[field, modified_field],
            )

            for r in rows:
                ext_val = r.get(field)
                if ext_val is None:
                    continue

                ext = str(ext_val)
                idx_mod = (
                    "" if r.get(modified_field) is None else str(r.get(modified_field))
                )
                inc_mod = incoming_map.get(ext)
                if inc_mod is None:
                    continue

                if idx_mod != inc_mod:
                    diffs.add(ext)

        return list(diffs)
