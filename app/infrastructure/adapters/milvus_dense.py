import contextlib
from typing import Any

from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer

from app.infrastructure.adapters.interfaces import IMilvusDense
from app.settings.config import Settings


class MilvusDense(IMilvusDense):
    """Упрощённый провайдер запроса в Milvus:
      - подключается к Milvus по host/port
      - кодирует запрос эмбеддером
      - вызывает search по коллекции
    Требования к коллекции:
      - vector_field: вектор признаков
      - id_field: внешний id документа
      - output_fields: ext_id, question, analysis, answer
    """

    def __init__(self, settings: Settings):
        self.app_model_name = settings.milvus_dense.app_model_name
        self.host = settings.milvus_dense.host
        self.port = settings.milvus_dense.port
        self.collection_name = settings.milvus_dense.collection
        self.vector_field = settings.milvus_dense.vector_field
        self.id_field = settings.milvus_dense.id_field
        self.output_fields = [f.strip() for f in settings.milvus_dense.output_fields.split(",")]
        # Подключение (ленивое)
        connections.connect(alias="default", host=self.host, port=str(self.port))
        self.col = Collection(self.collection_name)
        # Убедимся, что коллекция загружена в память (опционально)
        with contextlib.suppress(Exception):
            self.col.load()

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Поиск"""
        embedder = SentenceTransformer(self.app_model_name)
        vec = embedder.encode([query], normalize_embeddings=True)
        # параметры поиска можно настроить под тип индекса
        search_params = {"metric_type": "IP", "params": {"nprobe": 16}}
        res = self.col.search(
            data=vec,
            anns_field=self.vector_field,
            param=search_params,
            limit=top_k,
            output_fields=self.output_fields,
        )
        out: list[dict[str, Any]] = []
        if not res:
            return out
        hits = res[0]
        for h in hits:
            fields = h.entity.get(0)
            row = {k: fields.get(k) for k in self.output_fields}
            row["ext_id"] = row.get("ext_id") or fields.get(self.id_field)
            row["score_dense"] = float(h.distance)
            out.append(row)
        return out
