import time

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from app.infrastructure.adapters.interfaces import ICrossEncoderAdapter
from app.infrastructure.utils.metrics import metrics_print
from app.settings.config import Settings


class CrossEncoderAdapter(ICrossEncoderAdapter):
    """Адаптер кросс - энкодера"""

    def __init__(self, settings: Settings):
        ce_init_start = time.perf_counter()
        self.settings = settings
        self.model = CrossEncoder(
            settings.reranker.model_name, device=settings.reranker.device
        )
        metrics_print("🕒 Инициализация реранкера", ce_init_start)

    def rank(
        self, pairs: list[tuple[str:str]]
    ) -> list[torch.Tensor] | np.ndarray | torch.Tensor:
        """Реранжирование"""
        return self.model.predict(pairs)

    def rank_fast(
        self,
        pairs: list[tuple[str, str]],
        device: str = "cuda",
        batch_size: int = 128,
        max_length: int = 192,
        dtype: str = "fp16",
    ) -> list[float]:
        """Быстрое ранкирование"""
        import math
        import os

        import torch

        # не плодим лишний параллелизм токенайзера для маленьких партий
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        ce_model = self.model
        tok = getattr(ce_model, "tokenizer", None)
        mdl = getattr(ce_model, "model", None)

        if tok is None or mdl is None:
            return [float(x) for x in ce_model.predict(pairs)]

        # параметры из Settings
        device = self.settings.reranker.device or "cuda"
        batch_size = self.settings.reranker.batch_size or 128
        max_length = self.settings.reranker.max_length or 192
        dtype = self.settings.reranker.dtype or "fp16"

        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        mdl.eval().to(dev)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        if dtype == "fp16":
            amp_dtype = torch.float16
        elif dtype == "bf16":
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = None

        process_batches_start = time.perf_counter()
        out: list[float] = self._process_batches(
            pairs=pairs,
            mdl=mdl,
            tok=tok,
            max_length=max_length,
            batch_size=batch_size,
            dev=dev,
            amp_dtype=amp_dtype,
        )
        metrics_print("Encoder (process batches)", start_time=process_batches_start)

        clean = []
        for v in out:
            try:
                f = float(v)
                clean.append(f if math.isfinite(f) else 0.0)
            except (TypeError, ValueError):
                clean.append(0.0)
        return clean

    @staticmethod
    def _process_batches(
        pairs: list[tuple[str, str]],
        mdl: CrossEncoder,
        tok: CrossEncoder,
        max_length: int,
        batch_size: int,
        dev: torch.device,
        amp_dtype: torch.dtype | None = None,
    ) -> list[float]:
        out: list[float] = []
        with torch.inference_mode():
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i : i + batch_size]

                enc = tok(
                    [a for a, _ in batch],
                    [b for _, b in batch],
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}

                if amp_dtype and dev.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits = mdl(**enc).logits
                else:
                    logits = mdl(**enc).logits

                vals = logits.squeeze(-1).float().tolist()
                if isinstance(vals, float):
                    vals = [vals]
                out.extend(float(x) for x in vals)
        return out

    @staticmethod
    def _sigmoid(x: float) -> float:
        import math

        try:
            return 1.0 / (1.0 + math.exp(-float(x)))
        except Exception:
            return 0.5

    def ce_postprocess(self, logits: list[float]) -> list[float]:
        """Преобразует логиты CE в интерпретируемые очки.
        Режим управляется self.settings.ce_score_mode:
          - "sigmoid" (по умолчанию): независимая вероятность для каждого (query, doc)
          - "softmax": распределение по кандидатовому списку (с температурой)
        """
        mode = self.settings.reranker.score_mode
        if not logits:
            return []

        if mode == "softmax":
            import math

            temp = self.settings.reranker.softmax_temp
            m = max(logits)
            exps = [math.exp((x - m) / temp) for x in logits]
            z = sum(exps) or 1.0
            return [e / z for e in exps]

            # по умолчанию — сигмоида: логит -> вероятность
        return [self._sigmoid(x) for x in logits]
