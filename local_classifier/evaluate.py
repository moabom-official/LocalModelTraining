"""Evaluate the saved 3-class model on the 4-class test split with rejection.

3-class softmax 결과에 ``REJECTION_THRESHOLD`` 기반 rejection 을 적용해
4-class output 공간 (PO / VR / Q / NOISE) 의 정확도·P/R/F1·confusion matrix
를 계산한다. 추가로 다양한 임계값 sweep 으로 best tau 를 탐색.

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


def f1_per_class(y_true: list[str], y_pred: list[str], label: str) -> tuple[float, float, float, int]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
    support = tp + fn
    if tp + fp == 0 or tp + fn == 0:
        return 0.0, 0.0, 0.0, support
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return f1, prec, rec, support


def predict_with_rejection(
    max_probs: list[float],
    pred_ids: list[int],
    threshold: float,
) -> list[str]:
    """3-class softmax 결과를 4-class 출력 라벨로 변환."""
    out: list[str] = []
    for prob, pid in zip(max_probs, pred_ids):
        if prob < threshold:
            out.append(C.NOISE_LABEL)
        else:
            out.append(C.TRAINING_ID2LABEL[int(pid)])
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True

    model_path = C.MODEL_DIR / "best"
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}. Run train.py first.")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    n_train_labels = model.config.num_labels
    if n_train_labels != C.NUM_LABELS:
        print(
            f"[warn] model num_labels={n_train_labels} but config.NUM_LABELS={C.NUM_LABELS}; "
            f"expected 3-class head."
        )

    test_path = C.DATA_DIR / "test.jsonl"
    test_ds = CommentDataset(test_path, tokenizer, C.MAX_SEQ_LEN)
    loader = DataLoader(
        test_ds,
        batch_size=C.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=C.DATALOADER_NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    autocast_dtype = torch.bfloat16 if C.USE_BF16 else torch.float32
    all_max_probs: list[float] = []
    all_pred_ids: list[int] = []
    all_true_labels: list[str] = []

    # test.jsonl 의 label_id 는 4-class OUTPUT 공간. 실제 정답은 string label.
    raw_records: list[dict] = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))

    with torch.no_grad():
        idx = 0
        for batch in loader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            bsz = ids.size(0)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=device.type == "cuda"):
                logits = model(input_ids=ids, attention_mask=mask).logits
            probs = torch.softmax(logits.float(), dim=-1)
            max_p, pred = probs.max(dim=-1)
            all_max_probs.extend(max_p.cpu().tolist())
            all_pred_ids.extend(pred.cpu().tolist())
            for j in range(bsz):
                all_true_labels.append(raw_records[idx + j]["label"])
            idx += bsz

    # ----- 기본 임계값으로 보고 -----
    default_tau = C.REJECTION_THRESHOLD
    y_pred = predict_with_rejection(all_max_probs, all_pred_ids, default_tau)

    total = len(all_true_labels)
    correct = sum(1 for t, p in zip(all_true_labels, y_pred) if t == p)
    acc = correct / max(total, 1)
    print(f"\ntest acc = {acc:.4f}   n={total}   tau={default_tau}")
    print()
    f1s = []
    for label in C.OUTPUT_LABEL_NAMES:
        f1, prec, rec, sup = f1_per_class(all_true_labels, y_pred, label)
        f1s.append(f1)
        print(f"  {label:18s} support={sup:5d}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}")
    macro = sum(f1s) / len(C.OUTPUT_LABEL_NAMES)
    print(f"\nmacro F1 = {macro:.4f}")

    # 4-class confusion matrix
    print("\nconfusion matrix (rows=true, cols=pred):")
    header = " " * 14 + " ".join(f"{lbl[:6]:>7s}" for lbl in C.OUTPUT_LABEL_NAMES)
    print(header)
    cm: list[list[int]] = [[0] * len(C.OUTPUT_LABEL_NAMES) for _ in C.OUTPUT_LABEL_NAMES]
    for t, p in zip(all_true_labels, y_pred):
        cm[C.OUTPUT_LABEL2ID[t]][C.OUTPUT_LABEL2ID[p]] += 1
    for r, lbl in enumerate(C.OUTPUT_LABEL_NAMES):
        row = f"{lbl[:12]:12s}  " + " ".join(f"{cm[r][c]:7d}" for c in range(len(C.OUTPUT_LABEL_NAMES)))
        print(row)

    # ----- 임계값 sweep -----
    print()
    print("=" * 78)
    print("threshold sweep (4-class macro F1)")
    print("=" * 78)
    print(f"{'tau':>5s}  " + "  ".join(f"{lbl[:6]:>6s}" for lbl in C.OUTPUT_LABEL_NAMES) + "  macro")
    best_tau, best_macro = default_tau, macro
    for tau in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        y_sweep = predict_with_rejection(all_max_probs, all_pred_ids, tau)
        f1s_sweep = [f1_per_class(all_true_labels, y_sweep, l)[0] for l in C.OUTPUT_LABEL_NAMES]
        m = sum(f1s_sweep) / len(C.OUTPUT_LABEL_NAMES)
        marker = " <- best" if m > best_macro else ""
        print(f"{tau:>5.2f}  " + "  ".join(f"{f:>6.3f}" for f in f1s_sweep) + f"  {m:>6.3f}{marker}")
        if m > best_macro:
            best_macro = m
            best_tau = tau
    print(f"\nbest tau = {best_tau:.2f}   best macro F1 = {best_macro:.4f}")

    # ----- 산출물 -----
    out = C.LOG_DIR / "test_predictions.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i, (t, p, mp, pid) in enumerate(zip(all_true_labels, y_pred, all_max_probs, all_pred_ids)):
            f.write(json.dumps({
                "i": i,
                "true": t,
                "pred": p,
                "conf": round(float(mp), 4),
                "raw_pred_3class": C.TRAINING_ID2LABEL[int(pid)],
                "correct": t == p,
            }, ensure_ascii=False) + "\n")

    summary_path = C.LOG_DIR / "test_summary.json"
    summary_path.write_text(
        json.dumps({
            "acc": acc,
            "macro_f1": macro,
            "n": total,
            "tau": default_tau,
            "best_tau": best_tau,
            "best_macro_f1": best_macro,
            "cm": cm,
            "output_labels": C.OUTPUT_LABEL_NAMES,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nwrote {out}\nwrote {summary_path}")


if __name__ == "__main__":
    main()
