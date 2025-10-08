import numpy as np
import torch
from sentence_transformers import CrossEncoder

from app.infrastructure.adapters.interfaces import ICrossEncoderAdapter
from app.settings.config import Settings


class CrossEncoderAdapter(ICrossEncoderAdapter):
    """Адаптер кросс - энкодера"""

    def rank(self, model: CrossEncoder, pairs: list[tuple[str:str]]) -> list[torch.Tensor] | np.ndarray | torch.Tensor:
        """Реранжирование"""
        return model.predict(pairs)
