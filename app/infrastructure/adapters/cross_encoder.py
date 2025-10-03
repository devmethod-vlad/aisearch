import numpy as np
import torch, time
from sentence_transformers import CrossEncoder

from app.infrastructure.adapters.interfaces import ICrossEncoderAdapter
from app.settings.config import Settings


class CrossEncoderAdapter(ICrossEncoderAdapter):
    """Адаптер кросс - энкодера"""

    def __init__(self, settings: Settings):
        start_init = time.perf_counter()

        self.model = CrossEncoder(settings.reranker.model_name, device=settings.reranker.device)
        end_init = time.perf_counter()
        print(f"Время, затраченное на init CrossEncoderAdapter {start_init - end_init:.6f} секунд")

    def rank(self, pairs: list[tuple[str:str]]) -> list[torch.Tensor] | np.ndarray | torch.Tensor:
        """Реранжирование"""
        return self.model.predict(pairs)
