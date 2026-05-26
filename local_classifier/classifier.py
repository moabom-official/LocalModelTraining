"""LocalRobertaClassifier — inference-only wrapper.

Conforms to ``BaseCommentClassifier`` from
``comment_filtering_agent.classifiers.classifier_interface`` so the cascade
router can swap it for the existing API classifier with zero pipeline changes.

Outputs 4-class labels: PRODUCT_OPINION / VIDEO_REACTION / QUESTION / NOISE.
NOISE 는 양성 클래스로 직접 학습됨 (softmax + argmax 단순 분류).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from local_classifier import config as C

logger = logging.getLogger(__name__)


class LocalRobertaClassifier:
    def __init__(
        self,
        model_path: str | Path | None = None,
        use_gpu: bool = True,
        max_len: int = C.MAX_SEQ_LEN,
        use_bf16: bool = C.USE_BF16,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.model_path = Path(model_path or (C.MODEL_DIR / "best"))
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"model not found at {self.model_path}. Run local_classifier.train first."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device).eval()
        self.max_len = max_len
        self.use_bf16 = bool(use_bf16 and self.device.type == "cuda")
        self.stats = {
            "total_calls": 0,
            "avg_latency_ms": 0.0,
            "device": str(self.device),
            "bf16": self.use_bf16,
        }
        logger.info(
            "LocalRobertaClassifier loaded from %s on %s (bf16=%s)",
            self.model_path, self.device, self.use_bf16,
        )

    def _to_result(self, label_id: int, confidence: float, latency_ms: float):
        from comment_filtering_agent.classifiers.classifier_interface import (
            ClassificationResult,
        )
        label = C.ID2LABEL[int(label_id)]
        return ClassificationResult(
            label=label,
            confidence=float(confidence),
            rationale_short=f"local roberta (conf {confidence:.2f})",
            needs_recheck=(confidence < C.ROUTER_TAU_HIGH),
            mentioned_product_features=[],
            is_product_related=(label in {"PRODUCT_OPINION", "QUESTION"}),
            classifier_used="local-roberta",
            latency_ms=float(latency_ms),
        )

    def _predict(self, texts: list[str]) -> tuple[list[int], list[float]]:
        """Softmax 추론. 반환: (argmax 클래스 id, 그 클래스의 softmax 확률)."""
        torch = self._torch
        enc = self.tokenizer(
            list(texts),
            truncation=True,
            max_length=self.max_len,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        autocast_dtype = torch.bfloat16 if self.use_bf16 else torch.float32
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=autocast_dtype,
                                enabled=self.use_bf16):
                logits = self.model(**enc).logits
            probs = torch.softmax(logits.float(), dim=-1)
            conf, pred = probs.max(dim=-1)
        return pred.cpu().tolist(), conf.cpu().tolist()

    def classify(self, comment: str):
        t0 = time.time()
        preds, confs = self._predict([comment])
        lat = (time.time() - t0) * 1000
        self._record(lat, 1)
        return self._to_result(preds[0], confs[0], lat)

    def classify_batch(self, comments: list[str]):
        if not comments:
            return []
        t0 = time.time()
        preds, confs = self._predict(list(comments))
        lat = (time.time() - t0) * 1000
        per = lat / len(comments)
        self._record(lat, len(comments))
        return [self._to_result(p, c, per) for p, c in zip(preds, confs)]

    def _record(self, latency_ms: float, n: int) -> None:
        prev_n = self.stats["total_calls"]
        prev_avg = self.stats["avg_latency_ms"]
        new_n = prev_n + n
        self.stats["total_calls"] = new_n
        self.stats["avg_latency_ms"] = (
            (prev_avg * prev_n + latency_ms) / max(new_n, 1)
        )

    def get_stats(self) -> dict[str, Any]:
        return dict(self.stats)
