import os
from collections.abc import Iterable
from typing import Any

from whoosh import index
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, TEXT, Schema
from whoosh.qparser import MultifieldParser

from app.infrastructure.adapters.interfaces import IBM25WhooshAdapter
from app.settings.config import Settings


class BM25Adapter(IBM25WhooshAdapter):
    """Адаптер bm25 на whoosh"""

    def __init__(self, settings: Settings):
        self.index_path = settings.bm25.index_path
        self.schema_fields = settings.bm25.schema_fields
        self._ix = None
        self._ensure_index()

    def _ensure_index(self) -> None:
        os.makedirs(self.index_path, exist_ok=True)
        if not index.exists_in(self.index_path):
            schema = Schema(
                ext_id=ID(stored=True, unique=True),
                question=TEXT(stored=True, analyzer=StemmingAnalyzer()),
                analysis=TEXT(stored=True, analyzer=StemmingAnalyzer()),
                answer=TEXT(stored=True, analyzer=StemmingAnalyzer()),
            )
            self._ix = index.create_in(self.index_path, schema)
        else:
            self._ix = index.open_dir(self.index_path)

    def rebuild(self, documents: Iterable[dict[str, Any]]) -> None:
        """Полная пересборка индекса.
        documents: iterable словарей {"ext_id","question","analysis","answer"}
        """
        if self._ix is None:
            self._ensure_index()
        writer = self._ix.writer(limitmb=512, procs=1, multisegment=True)
        try:
            for doc in documents:
                writer.update_document(
                    ext_id=str(doc.get("ext_id", "")),
                    question=str(doc.get("question", "")),
                    analysis=str(doc.get("analysis", "")),
                    answer=str(doc.get("answer", "")),
                )
        finally:
            writer.commit()

    def search(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
        """Возвращает список кандидатов:
        [{"ext_id","question","analysis","answer","score_bm25","source":"bm25"}, ...]
        """
        if self._ix is None:
            self._ensure_index()
        fields = [f for f in self.schema_fields if f in ("question", "analysis", "answer")]
        parser = MultifieldParser(fields, schema=self._ix.schema)
        q = parser.parse(query)
        results: list[dict[str, Any]] = []
        with self._ix.searcher() as s:
            hits = s.search(q, limit=top_k)
            for h in hits:
                results.append(
                    {
                        "ext_id": h.get("ext_id"),
                        "question": h.get("question"),
                        "analysis": h.get("analysis"),
                        "answer": h.get("answer"),
                        "score_bm25": float(h.score),
                        "source": "bm25",
                    }
                )
