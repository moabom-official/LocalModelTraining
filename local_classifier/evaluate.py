"""Evaluate the saved best 4-class model on the test split.

Reports overall accuracy, per-class precision/recall/F1, macro-F1, and a
confusion matrix. Also writes per-comment predictions for error analysis.

Run:  python -m local_classifier.evaluate
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from local_classifier import config as C
from local_classifier.dataset import CommentDataset


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"BASE_MODEL = {C.BASE_MODEL}")
    model_path = C.MODEL_DIR / "best"
    print(f"model_path = {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}. Run train.py first.")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    test_path = C.DATA_DIR / "test.jsonl"
    test_ds = CommentDataset(test_path, tokenizer, C.MAX_SEQ_LEN)
    loader = DataLoader(
        test_ds,
        batch_size=C.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=C.DATALOADER_NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    n = C.NUM_LABELS
    cm = [[0] * n for _ in range(n)]
    correct, total = 0, 0
    per_pred_logs: list[dict] = []
    autocast_dtype = torch.bfloat16 if C.USE_BF16 else torch.float32

    with torch.no_grad():
        offset = 0
        for batch in loader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=device.type == "cuda"):
                logits = model(input_ids=ids, attention_mask=mask).logits
            probs = torch.softmax(logits.float(), dim=-1)
            conf, preds = probs.max(dim=-1)
            for t, p, k in zip(labels.tolist(), preds.tolist(), conf.tolist()):
                cm[t][p] += 1
                per_pred_logs.append({
                    "i": offset,
                    "true": C.ID2LABEL[t],
                    "pred": C.ID2LABEL[p],
                    "conf": round(k, 4),
                    "correct": t == p,
                })
                offset += 1
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / max(total, 1)
    print(f"\ntest acc = {acc:.4f}   n={total}\n")

    f1s = []
    for c in range(n):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(n) if r != c)
        fn = sum(cm[c][r] for r in range(n) if r != c)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        f1s.append(f1)
        print(f"  {C.ID2LABEL[c]:18s} support={tp+fn:5d}  "
              f"prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}")
    macro = sum(f1s) / n
    print(f"\nmacro F1 = {macro:.4f}")

    print("\nconfusion matrix (rows=true, cols=pred):")
    header = " " * 14 + " ".join(f"{C.ID2LABEL[i][:6]:>7s}" for i in range(n))
    print(header)
    for r in range(n):
        row = f"{C.ID2LABEL[r][:12]:12s}  " + " ".join(f"{cm[r][c]:7d}" for c in range(n))
        print(row)

    out = C.LOG_DIR / "test_predictions.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in per_pred_logs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = C.LOG_DIR / "test_summary.json"
    summary.write_text(
        json.dumps({"acc": acc, "macro_f1": macro, "n": total, "cm": cm,
                    "labels": C.LABEL_NAMES},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nwrote {out}\nwrote {summary}")


if __name__ == "__main__":
    main()
