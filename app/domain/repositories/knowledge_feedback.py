from app.common.repositories.repository import SQLAlchemyRepository
from app.domain.repositories.interfaces import IKnowledgeFeedbackRepository
from app.domain.schemas.knowledge_feedback import KnowledgeFeedbackSchema
from app.infrastructure.models.knowledge_feedback import KnowledgeFeedback


class KnowledgeFeedbackRepository(SQLAlchemyRepository, IKnowledgeFeedbackRepository):
    """Репозиторий оценки знания"""

    model = KnowledgeFeedback
    response_dto = KnowledgeFeedbackSchema
