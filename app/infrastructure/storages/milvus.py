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
    MilvusException,
)
from sentence_transformers import SentenceTransformer

from app.common.logger import AISearchLogger
from app.infrastructure.storages.interfaces import IVectorDatabase
from app.infrastructure.utils.metrics import metrics_print
from app.infrastructure.utils.milvus import load_schema_and_indexes_from_json
from app.infrastructure.utils.nlp import l2_normalize
from app.infrastructure.utils.universal import async_retry
from app.settings.config import MilvusSettings


class MilvusDatabase(IVectorDatabase):
    """Класс для работы с Milvus DB с использованием AsyncMilvusClient."""

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

        # Только предзагрузка коллекций (metadata логика удалена)

        metrics_print("🕒 Инициализация Milvus", milvus_init_start)

    @staticmethod
    def get_model_name(model: SentenceTransformer) -> str:
        """Получить имя модели"""
        return model._first_module().auto_model.config._name_or_path.split("/")[-1]

    async def load_collection(self, collection_name: str) -> None:
        """Подгрузка коллекции"""
        await self.client.load_collection(
            collection_name, timeout=self.config.query_timeout
        )
        self.__collections_loaded.add(collection_name)

    async def create_collection(
        self,
        collection_name: str,
    ) -> None:
        """Создает коллекцию для хранения векторов"""
        collections = await self.client.list_collections(
            timeout=self.config.query_timeout
        )
        if collection_name in collections:
            await self.client.drop_collection(
                collection_name, timeout=self.config.query_timeout
            )

        fields, index_specs, search_params_by_field = load_schema_and_indexes_from_json(
            self.config.schema_path
        )

        schema = CollectionSchema(fields, description=f"Collection {collection_name}")
        await self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            timeout=self.config.query_timeout,
        )

        if index_specs:
            field_names = {f.name for f in fields}
            index_params = self.client.prepare_index_params()

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

            await self.client.create_index(
                collection_name=collection_name,
                index_params=index_params,
                timeout=self.config.query_timeout,
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
        """Вставка векторов и метаданных с проверкой типов и размерности."""
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

        # dim может быть как атрибутом .dim, так и в params (в разных версиях pymilvus)
        vfs = f_by_name[vec_field]
        dim = getattr(vfs, "dim", None)
        if dim is None:
            params = getattr(vfs, "params", {}) or {}
            dim = params.get("dim")

        # 2) Нормализация одного вектора (тип и размерность)
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
            """-> (coerced_value, drop)
            drop=True означает 'не включать это поле в запись'
            """
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

            # Готовим данные построчно: {vector_field: [...], **coerced_meta}
            data: list[dict[str, tp.Any]] = []
            for vec, row in zip(vectors_batch, rows_batch, strict=True):
                item: dict[str, tp.Any] = {}

                # вектор
                item[vec_field] = _normalize_vector(vec)

                # метаданные (только поля из схемы, без None)
                for k, v in row.items():
                    val, drop = _coerce(k, v)
                    if not drop:
                        item[k] = val

                data.append(item)

            # вставка батча
            await self.client.insert(
                collection_name=collection_name,
                data=data,
                timeout=self.config.query_timeout,
            )
            self.logger.info(f"Загружено {i + 1}/{num_batches} батчей")

        # 5) Финальный flush (один раз, после всех батчей)
        await self.client.flush(collection_name, timeout=self.config.query_timeout)

    async def search(
        self, collection_name: str, query_vector: list[float], top_k: int
    ) -> list[dict[str, tp.Any]]:
        """Поиск по косинусной схожести."""
        top_k = max(top_k, 1)

        if collection_name not in self.__collections_loaded:
            await self.load_collection(collection_name)

        results = await self.client.search(
            collection_name=collection_name,
            data=[query_vector],
            anns_field=self.config.vector_field,
            params={"metric_type": self.config.metric_type, "params": {"ef": 64}},
            limit=top_k,
            output_fields=self.config.output_fields,
            timeout=self.config.query_timeout,
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
        """Проверка наличия и готовности коллекции."""
        try:
            collections = await self.client.list_collections(
                timeout=self.config.query_timeout
            )
            if collection_name not in collections:
                return False

            indexes = await self.client.list_indexes(collection_name)
            return len(indexes) > 0
        except Exception:
            return False

    async def delete_collection(self, collection_name: str) -> None:
        """Удаление коллекции."""
        self.logger.info(f"Удаление коллекции {collection_name} ...")

        collections = await self.client.list_collections(
            timeout=self.config.query_timeout
        )
        if collection_name not in collections:
            self.logger.info(
                f"Коллекция {collection_name} не существует, удаление не требуется."
            )
            self.__collections_loaded.discard(collection_name)
            return

        await self.client.drop_collection(
            collection_name, timeout=self.config.query_timeout
        )
        self.__collections_loaded.discard(collection_name)
        self.logger.info(f"Коллекция {collection_name} успешно удалена")

    @async_retry(max_attempts=5, delay=3, exceptions=(MilvusException,))
    async def safe_delete_collection(self, collection_name: str) -> None:
        """Удаление коллекции с попытками"""
        await self.delete_collection(collection_name)

    async def preload_collections(self) -> None:
        """Предзагрузка коллекций в память"""
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
        """Индексация документов в vector_db."""
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
        """Инициализация коллекции с текущей моделью."""
        self.logger.info(f"Инициализация коллекции {collection_name} ...")
        await self.index_documents(
            collection_name=collection_name,
            model=model,
            documents=documents,
            metadata=metadata,
        )

    async def close(self) -> None:
        """Закрытие соединения с клиентом."""
        if hasattr(self, "client"):
            await self.client.close()

    async def fetch_existing(
        self, collection_name: str, output_fields: list[str] | None = None
    ) -> list[dict]:
        """Получить все данные из коллекции пакетами по 8_192, корректно для auto_id и строковых полей"""
        try:
            if collection_name not in self.__collections_loaded:
                await self.load_collection(collection_name)

            output_fields = (
                output_fields or self.config.output_fields.split(",")
                if isinstance(self.config.output_fields, str)
                else self.config.output_fields
            )
            # row_count = int(
            #     (await self.client.get_collection_stats(collection_name))["row_count"]
            # )
            results = []

            batch_size = 8_192
            last_pk = -1

            while True:
                # Берём пакет записей по автоинкрементному PK
                filter_expr = f"pk > {last_pk}"
                batch_res = await self.client.query(
                    collection_name=collection_name,
                    filter=filter_expr,
                    output_fields=output_fields,
                    limit=batch_size,
                    timeout=self.config.query_timeout,
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
        """Вставка/обновление векторов и метаданных (upsert по ext_id)."""
        if not metadata:
            metadata = [{} for _ in vectors]
        elif len(metadata) != len(vectors):
            raise ValueError(
                "Количество векторов не совпадает с количеством метаданных"
            )

        # 1) Получаем существующие записи по ext_id
        existing = await self.fetch_existing(
            collection_name, output_fields=["ext_id", "pk"]
        )
        ext_id_to_pk = {r["ext_id"]: r["pk"] for r in existing}

        # 2) Определяем pk, которые нужно удалить (существующие)
        pk_to_delete = [
            ext_id_to_pk[m["ext_id"]] for m in metadata if m["ext_id"] in ext_id_to_pk
        ]
        if pk_to_delete:
            # В Milvus нет delete_by_ids для auto_id, используем filter
            filter_expr = "pk in [" + ",".join(map(str, pk_to_delete)) + "]"
            await self.client.delete(
                collection_name=collection_name, filter=filter_expr
            )
            await self.client.flush(collection_name)

        # 3) Подготовка данных
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

        def _coerce(name: str, value: tp.Any) -> tuple[tp.Any | bool]:
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

        # 4) Вставка батчами
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
                item = {vec_field: vec}
                for k, v in row.items():
                    val, drop = _coerce(k, v)
                    if not drop:
                        item[k] = val
                data.append(item)

            await self.client.insert(
                collection_name=collection_name,
                data=data,
                timeout=self.config.query_timeout,
            )
            self.logger.info(f"Загружено {i + 1}/{num_batches} батчей")

        await self.client.flush(collection_name, timeout=self.config.query_timeout)
        self.logger.info("Upsert завершен ✅")

    async def delete_vectors(
        self,
        collection_name: str,
        ext_ids: list[str] | None = None,
        filter_expr: str | None = None,
    ) -> None:
        """Удаляет записи из коллекции Milvus.

        Можно удалить:
          - по списку ext_id (list[str])
          - или по произвольному фильтру (filter_expr)
        """
        if not ext_ids and not filter_expr:
            raise ValueError("Нужно передать либо ext_ids, либо filter_expr")

        # Загружаем коллекцию, если ещё не загружена
        if collection_name not in self.__collections_loaded:
            await self.load_collection(collection_name)

        # Если передан список ext_id — формируем выражение фильтра
        if ext_ids:
            quoted_ids = ",".join(f"'{x}'" for x in ext_ids)
            filter_expr = f"ext_id in [{quoted_ids}]"

        self.logger.info(
            f"🧹 Удаление записей из {collection_name} по фильтру: {filter_expr}"
        )

        try:
            await self.client.delete(
                collection_name=collection_name,
                filter=filter_expr,
                timeout=self.config.query_timeout,
            )
            await self.client.flush(collection_name, timeout=self.config.query_timeout)
            self.logger.info(f"✅ Удаление завершено ({collection_name})")
        except Exception as e:
            self.logger.error(
                f"Ошибка при удалении из {collection_name} ({type(e)}): {traceback.format_exc()}"
            )

    async def collection_not_empty(self, collection_name: str) -> bool:
        """Проверка, что коллекция существует и содержит хотя бы 1 запись через fetch_existing"""
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
        """Проверяем, что поле существует и имеет тип VarChar."""
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
        """Возвращает:
        - found_incoming: входящие ext_id, найденные в коллекции
        - missing_incoming: входящие ext_id, которых нет в коллекции
        - extra_in_store: ext_id из коллекции, отсутствующие во входящих

        Если заданы source_field и source, все операции выполняются
        только по документам, у которых source_field == source.
        """
        # ------------------------------------
        # 0. Проверка схемы коллекции
        # ------------------------------------
        schema = await self.client.describe_collection(collection_name)
        fields = {f["name"]: f for f in schema["fields"]}

        if field not in fields:
            raise RuntimeError(
                f"Field '{field}' not found in collection '{collection_name}'"
            )

        if fields[field]["type"] != DataType.VARCHAR:
            raise RuntimeError(
                f"Field '{field}' must be VarChar, got {fields[field]['type']}"
            )

        # Обязательность пары source_field / source
        if (source is None) != (source_field is None):
            raise ValueError(
                "Both 'source' and 'source_field' must be provided together or omitted"
            )

        # Проверка поля source_field, если фильтрация по source включена
        if source is not None and source_field is not None:
            if source_field not in fields:
                raise RuntimeError(
                    f"Field '{source_field}' not found in collection '{collection_name}'"
                )

            if fields[source_field]["type"] != DataType.VARCHAR:
                raise RuntimeError(
                    f"Field '{source_field}' must be VarChar, got {fields[source_field]['type']}"
                )

        # ------------------------------------
        # 1. Load коллекции
        # ------------------------------------
        with contextlib.suppress(Exception):
            await self.load_collection(collection_name)

        # ------------------------------------
        # 2. Приведение входящих в строки
        # ------------------------------------
        incoming_ids = [str(x) for x in incoming_ext_ids if x is not None]
        incoming_set = set(incoming_ids)
        found_incoming: set[str] = set()

        # экранированное значение source (если нужно)
        escaped_source_value: str | None = None
        if source is not None:
            escaped_source_value = self._escape_str_for_expr(source)

        # ------------------------------------
        # 3. Поиск входящих батчами
        # ------------------------------------
        for i in range(0, len(incoming_ids), batch_size):
            batch = incoming_ids[i : i + batch_size]

            list_literal = ",".join(f'"{self._escape_str_for_expr(s)}"' for s in batch)
            base_expr = f"{field} in [{list_literal}]"

            if escaped_source_value is not None and source_field is not None:
                # фильтрация и по ext_id, и по source_field
                filter_expr = (
                    f"({base_expr}) AND {source_field} == " f'"{escaped_source_value}"'
                )
            else:
                filter_expr = base_expr

            res = await self.client.query(
                collection_name=collection_name,
                filter=filter_expr,
                output_fields=[field],
                offset=0,
                limit=batch_size,
                timeout=30.0,
            )

            for r in res:
                v = r.get(field)
                if v is not None:
                    found_incoming.add(str(v))

        missing_incoming = incoming_set - found_incoming

        # ------------------------------------
        # 4. extra_in_store — через pagination (offset+limit)
        # ------------------------------------
        extra_in_store: set[str] = set()
        store_ids: set[str] = set()

        # фильтр по source_field, если задан
        if escaped_source_value is not None and source_field is not None:
            extras_filter_expr = f'{source_field} == "{escaped_source_value}"'
        else:
            extras_filter_expr = ""  # всё

        offset = 0
        while True:
            res = await self.client.query(
                collection_name=collection_name,
                filter=extras_filter_expr,
                output_fields=[field],
                offset=offset,
                limit=iterator_batch,
                timeout=30.0,
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
        """Удаляет entities по строковому полю (VarChar) батчами.
        Возвращает количество удалённых entities.
        """
        if not ext_ids:
            return 0

        # ----------------------------
        # 1. Приведение incoming ext_ids к строкам
        # ----------------------------
        wanted = [str(x) for x in ext_ids if x is not None]
        if not wanted:
            return 0

        deleted_total = 0

        # ----------------------------
        # 2. Батчевое удаление
        # ----------------------------
        for i in range(0, len(wanted), batch_size):
            batch = wanted[i : i + batch_size]

            list_literal = ",".join(f'"{self._escape_str_for_expr(v)}"' for v in batch)
            filter_expr = f"{field} in [{list_literal}]"

            try:
                mr = await self.client.delete(
                    collection_name=collection_name,
                    filter=filter_expr,
                    timeout=30.0,
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
        """Возвращает список ext_id, у которых modified_at в Milvus отличается
        от входящего значения. Оба поля должны быть VarChar.
        """
        # ----------------------------
        # 1. Проверка схемы коллекции
        # ----------------------------
        schema = await self.client.describe_collection(collection_name)
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

        # ----------------------------
        # 2. Load коллекции
        # ----------------------------
        with contextlib.suppress(Exception):
            await self.load_collection(collection_name)

        # ----------------------------
        # 3. Приведение входящих значений
        # ----------------------------
        incoming_map = {
            str(k): ("" if v is None else str(v)) for k, v in incoming_modified.items()
        }
        ids = list(incoming_map.keys())

        diffs: set[str] = set()

        # ----------------------------
        # 4. Batch-query с filter
        # ----------------------------
        for i in range(0, len(ids), batch_size):
            batch = ids[i : i + batch_size]

            list_literal = ",".join(f'"{self._escape_str_for_expr(x)}"' for x in batch)
            filter_expr = f"{field} in [{list_literal}]"

            rows = await self.client.query(
                collection_name=collection_name,
                filter=filter_expr,
                output_fields=[field, modified_field],
                timeout=30.0,
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
