from app.infrastructure.models.base import Base
from app.infrastructure.models.component import Component
from app.infrastructure.models.product import Product
from app.infrastructure.models.role import Role
from app.infrastructure.models.search_entry import SearchEntry
from app.infrastructure.models.search_request import SearchRequest
from app.infrastructure.models.search_feedback import SearchFeedback
from app.infrastructure.models.knowledge_feedback import KnowledgeFeedback
from app.infrastructure.models.glossary_element import GlossaryElement
from app.infrastructure.models.source import Source

models = [
    Base,
    SearchRequest,
    SearchFeedback,
    KnowledgeFeedback,
    Component,
    Product,
    Role,
    SearchEntry,
    Source,
    GlossaryElement,
]
