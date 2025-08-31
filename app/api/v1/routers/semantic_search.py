from dishka import FromDishka
from dishka.integrations.fastapi import inject
from fastapi import APIRouter

from app.api.v1.dto.requests.semantic_search import ExampleRequest
from app.api.v1.dto.responses.semantic_search import ExampleResponse
from app.api.v1.dto.responses.taskmanager import (
    TaskQueryResponse,
)
from app.services.interfaces import ISemanticSearchService

router = APIRouter(prefix="/semantic-search", tags=["Semantic Search"])


@router.post(
    "/example",
    response_model=TaskQueryResponse | list[ExampleResponse],
    summary="Пример семантического поиска",
)
@inject
async def example(
    service: FromDishka[ISemanticSearchService], request: ExampleRequest
) -> TaskQueryResponse | list[ExampleResponse]:
    """Пример семантического поиска."""
    return await service.search(
        collection_name="example_collection",
        query="Какие языки программирования используются для машинного обучения?",
        top_k=request.top_k,
    )
