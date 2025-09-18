from opensearchpy import OpenSearch

from app.infrastructure.adapters.interfaces import IOpenSearchAdapter
from app.settings.config import Settings


class OpenSearchAdapter(IOpenSearchAdapter):
    """Адаптер для opensearch"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenSearch(
            hosts=[{"host": settings.opensearch.host, "port": settings.opensearch.port}],
            http_compress=True,
            http_auth=(
                (settings.opensearch.user, settings.opensearch.password)
                if settings.opensearch.user
                else None
            ),
            use_ssl=settings.opensearch.use_ssl,
            verify_certs=settings.opensearch.verify_certs,
        )

    def search(self, body: dict, size: int) -> list[dict]:
        """Поиск при помощи opensearch"""
        resp = self.client.search(index=self.settings.opensearch.index_name, body=body, size=size)
        return resp["hits"]["hits"]
