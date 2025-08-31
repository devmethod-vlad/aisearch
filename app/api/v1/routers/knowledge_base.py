from dishka import FromDishka
from dishka.integrations.fastapi import inject
from fastapi import APIRouter

from app.api.v1.dto.responses.knowledge_base import CollectDataResponse
from app.api.v1.dto.responses.taskmanager import TaskQueryResponse
from app.services.interfaces import IKnowledgeBaseService

router = APIRouter(prefix="/knowledge-base", tags=["Knowledge Base"])


@router.get(
    "/collect/{page_id}",
    response_model=CollectDataResponse,
    summary="Сбор данных БЗ из страницы Confluence",
)
@inject
async def collect_by_page_id(
    page_id: str, service: FromDishka[IKnowledgeBaseService]
) -> CollectDataResponse:
    """Сбор данных БЗ из страницы Confluence"""
    return await service.collect_data_from_confluence_by_page_id(page_id=page_id)


@router.get(
    "/collect/detached/{page_id}",
    response_model=TaskQueryResponse,
    summary="Сбор данных БЗ из страницы Confluence в фоновом режиме",
)
@inject
async def collect_by_page_id_detached(
    page_id: str, service: FromDishka[IKnowledgeBaseService]
) -> TaskQueryResponse:
    """Сбор данных БЗ из страницы Confluence"""
    return await service.collect_data_from_confluence_by_page_id_detached(page_id=page_id)


@router.get(
    "/collect-all",
    response_model=CollectDataResponse,
    summary="Сбор данных БЗ из страницы Confluence",
)
@inject
async def collect_all(service: FromDishka[IKnowledgeBaseService]) -> CollectDataResponse:
    """Сбор всех данных БЗ"""
    return await service.collect_all_data_from_confluence()


@router.get(
    "/collect-all/detached",
    response_model=TaskQueryResponse,
    summary="Сбор данных БЗ из страницы Confluence в фоновом режиме",
)
@inject
async def collect_all_detached(service: FromDishka[IKnowledgeBaseService]) -> TaskQueryResponse:
    """Сбор всех данных БЗ"""
    return await service.collect_all_data_from_confluence_detached()
