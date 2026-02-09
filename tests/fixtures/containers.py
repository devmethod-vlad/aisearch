import typing as tp

import pytest
from testcontainers.milvus import MilvusContainer
from testcontainers.opensearch import OpenSearchContainer


@pytest.fixture(scope="session")
def milvus_container() -> tp.Generator[MilvusContainer, None, None]:
    """Контейнер Milvus для всей сессии тестов"""
    with MilvusContainer("milvusdb/milvus:v2.5.12") as container:
        yield container


@pytest.fixture(scope="session")
def opensearch_container() -> tp.Generator[OpenSearchContainer, None, None]:
    """Контейнер OpenSearch для всей сессии тестов"""
    with OpenSearchContainer(image="opensearchproject/opensearch:2.12.0") as container:
        container.get_client()
        yield container
