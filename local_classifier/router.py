"""CascadeRouter — local RoBERTa first, GPT-4.1 API fallback for low conf.

  conf >= tau_high  → accept local
  tau_low <= conf < tau_high → escalate to API
  conf < tau_low → escalate AND log as Disagreement candidate

The router accepts any object implementing ``classify_batch`` and returning
objects with ``label`` / ``confidence`` / ``classifier_used`` attributes
(i.e. ``BaseCommentClassifier`` from the comment_filtering_agent module).
"""
from __future__ import annotations

import logging
from typing import Any

from local_classifier import config as C

logger = logging.getLogger(__name__)


class CascadeRouter:
    def __init__(
        self,
        local,
        api,
        tau_high: float = C.ROUTER_TAU_HIGH,
        tau_low: float = C.ROUTER_TAU_LOW,
    ) -> None:
        if not (0.0 <= tau_low <= tau_high <= 1.0):
            raise ValueError(
                f"thresholds must satisfy 0 <= tau_low <= tau_high <= 1, "
                f"got tau_low={tau_low}, tau_high={tau_high}"
            )
        self.local = local
        self.api = api
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.stats = {
            "total": 0,
            "local_accepted": 0,
            "api_escalated": 0,
            "low_conf_escalated": 0,
        }

    def classify(self, comment: str):
        return self.classify_batch([comment])[0]

    def classify_batch(self, comments: list[str]):
        if not comments:
            return []

        local_results = self.local.classify_batch(comments)
        finals: list[Any] = [None] * len(comments)
        escalate_idx: list[int] = []

        for i, r in enumerate(local_results):
            if r.confidence >= self.tau_high:
                r.classifier_used = "local-accept"
                finals[i] = r
                self.stats["local_accepted"] += 1
            else:
                escalate_idx.append(i)
                if r.confidence < self.tau_low:
                    self.stats["low_conf_escalated"] += 1

        if escalate_idx:
            api_results = self.api.classify_batch([comments[i] for i in escalate_idx])
            for j, i in enumerate(escalate_idx):
                api_r = api_results[j]
                api_r.classifier_used = "api-fallback"
                finals[i] = api_r
                self.stats["api_escalated"] += 1
                logger.info(
                    "router escalate: local=%s(%.2f) → api=%s(%.2f) | %s",
                    local_results[i].label,
                    local_results[i].confidence,
                    api_r.label,
                    getattr(api_r, "confidence", 0.0),
                    comments[i][:80].replace("\n", " "),
                )

        self.stats["total"] += len(comments)
        return finals

    def get_stats(self) -> dict[str, Any]:
        s = dict(self.stats)
        n = max(s["total"], 1)
        s["local_rate"] = s["local_accepted"] / n
        s["api_rate"] = s["api_escalated"] / n
        s["low_conf_rate"] = s["low_conf_escalated"] / n
        if hasattr(self.local, "get_stats"):
            s["local_stats"] = self.local.get_stats()
        if hasattr(self.api, "get_stats"):
            s["api_stats"] = self.api.get_stats()
        return s
