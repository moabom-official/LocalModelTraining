"""ShadowLogger — run local model in parallel with the production API
classifier, log disagreements, return the API result (authoritative).

Use this in P2 of the roadmap (shadow mode) to gather data for picking
``ROUTER_TAU_HIGH`` and ``ROUTER_TAU_LOW`` from real traffic without
risking accuracy regressions.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from local_classifier import config as C

logger = logging.getLogger(__name__)


class ShadowLogger:
    def __init__(
        self,
        api_classifier,
        local_classifier,
        log_path: str | Path | None = None,
    ) -> None:
        self.api = api_classifier
        self.local = local_classifier
        self.log_path = Path(log_path or (C.LOG_DIR / "shadow.jsonl"))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.stats = {"total": 0, "agree": 0, "disagree": 0}

    def classify(self, comment: str):
        return self.classify_batch([comment])[0]

    def classify_batch(self, comments: list[str]):
        if not comments:
            return []

        t0 = time.time()
        api_results = self.api.classify_batch(comments)
        api_total_ms = (time.time() - t0) * 1000

        try:
            t1 = time.time()
            local_results = self.local.classify_batch(comments)
            local_total_ms = (time.time() - t1) * 1000
        except Exception as e:  # noqa: BLE001 — shadow must never break prod
            logger.warning("shadow local failed (returning API only): %s", e)
            return api_results

        n = len(comments)
        with open(self.log_path, "a", encoding="utf-8") as f:
            for c, a, l in zip(comments, api_results, local_results):
                agree = a.label == l.label
                self.stats["total"] += 1
                self.stats["agree" if agree else "disagree"] += 1
                f.write(json.dumps({
                    "ts": time.time(),
                    "text": c[:500],
                    "api_label": a.label,
                    "api_conf": getattr(a, "confidence", None),
                    "local_label": l.label,
                    "local_conf": l.confidence,
                    "agree": agree,
                    "api_ms": api_total_ms / n,
                    "local_ms": local_total_ms / n,
                }, ensure_ascii=False) + "\n")
        return api_results

    def get_stats(self) -> dict[str, Any]:
        s = dict(self.stats)
        n = max(s["total"], 1)
        s["agreement_rate"] = s["agree"] / n
        return s
