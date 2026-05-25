"""End-to-end dataset prep (3-class training + rejection scheme):

  comment_labels/labeled_gpt41_azure.jsonl
    ↓ filter (teacher, confidence, len) — 4-class membership (PO/VR/Q/NOISE)
    ↓ normalize (NFKC, PII scrub, repeat-char compress)
    ↓ language detect
    ↓ near-dup dedup
    ↓ video_id grouped split (train / val / test)
    ↓ train/val: drop NOISE records (model learns only 3 positive classes)
    ↓ test:      keep all 4 classes (rejection 평가 가능)
    ↓ class weights (3-class)
  artifacts/data/{train,val,test}.jsonl  +  class_weights.json

라벨 ID 스페이스가 split 별로 다름:
- train.jsonl / val.jsonl : label_id ∈ {0,1,2}  (TRAINING_LABEL2ID)
- test.jsonl              : label_id ∈ {0,1,2,3}  (OUTPUT_LABEL2ID, NOISE=3)
이는 학습 시 NUM_LABELS=3 모델 head 와 정합. 추론 후 rejection 거쳐 4-class
output 공간으로 mapping.

Run:  python -m local_classifier.prepare_dataset
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from local_classifier import config as C
from local_classifier.preprocess import detect_lang, near_dup_key, normalize_text


def load_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def clean_record(rec: dict) -> dict | None:
    if rec.get("teacher_model") in C.DROP_TEACHERS:
        return None
    # 구 5-class 라벨(CHATTER / OFF_TOPIC)은 자동으로 NOISE 로 통합.
    label = C.remap_legacy_label(rec.get("label"))
    # 4-class (PO/VR/Q/NOISE) membership 체크. label_id 는 split 에서
    # train/val 용으로는 TRAINING_LABEL2ID 로 재매핑됨.
    if label not in C.OUTPUT_LABEL2ID:
        return None
    try:
        conf = float(rec.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return None
    if conf < C.MIN_CONFIDENCE:
        return None
    raw_text = rec.get("text") or ""
    text = normalize_text(raw_text)
    if len(text) < C.MIN_TEXT_LEN or len(text) > C.MAX_TEXT_LEN:
        return None
    return {
        "comment_id": rec.get("comment_id"),
        "video_id": rec.get("video_id"),
        "product_id": rec.get("product_id"),
        "text": text,
        "label": label,
        # 4-class id (test/eval 용). train/val 은 write 직전에 3-class 로 재매핑.
        "label_id": C.OUTPUT_LABEL2ID[label],
        "confidence": conf,
        "lang": detect_lang(text),
    }


def video_grouped_split(records: list[dict], val_ratio: float, test_ratio: float, seed: int):
    by_video: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_video[r["video_id"]].append(r)
    videos = sorted(by_video.keys())
    rng = random.Random(seed)
    rng.shuffle(videos)
    n = len(videos)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    test_v = set(videos[:n_test])
    val_v = set(videos[n_test:n_test + n_val])
    train, val, test = [], [], []
    for v, recs in by_video.items():
        bucket = test if v in test_v else val if v in val_v else train
        bucket.extend(recs)
    return train, val, test


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def show_stats(name: str, records: list[dict]) -> None:
    labels = Counter(r["label"] for r in records)
    langs = Counter(r["lang"] for r in records)
    print(f"\n[{name}] n={len(records)}")
    print(f"  labels: {dict(labels)}")
    print(f"  langs : {dict(langs)}")


def main() -> None:
    if not C.INPUT_JSONL.exists():
        raise FileNotFoundError(f"input not found: {C.INPUT_JSONL}")

    raw = list(load_jsonl(C.INPUT_JSONL))
    print(f"raw records: {len(raw)}")

    # 구 5-class → 4-class 자동 매핑 통계
    raw_label_counts = Counter(rec.get("label") for rec in raw)
    remapped = sum(
        raw_label_counts.get(legacy, 0) for legacy in C.LEGACY_LABEL_REMAP
    )
    if remapped:
        print(
            f"legacy label remap: "
            + ", ".join(
                f"{legacy}({raw_label_counts.get(legacy, 0)})"
                for legacy in C.LEGACY_LABEL_REMAP
            )
            + f" → NOISE ({remapped} records)"
        )

    cleaned: list[dict] = []
    seen: set[str] = set()
    drop = Counter()
    for rec in raw:
        c = clean_record(rec)
        if c is None:
            drop["clean"] += 1
            continue
        key = near_dup_key(c["text"])
        if key in seen:
            drop["dup"] += 1
            continue
        seen.add(key)
        cleaned.append(c)
    print(f"cleaned: {len(cleaned)}  dropped: {dict(drop)}")

    train, val, test = video_grouped_split(
        cleaned, C.VAL_RATIO, C.TEST_RATIO, C.SEED
    )

    # ----- 3-class training + rejection scheme -----
    # train/val 에서 NOISE 제거 + label_id 를 TRAINING_LABEL2ID 공간으로 재매핑.
    # test 는 NOISE 포함 + 4-class label_id 유지 (rejection 평가 용).
    def _to_training_record(r: dict) -> dict:
        return {**r, "label_id": C.TRAINING_LABEL2ID[r["label"]]}

    train_kept = [_to_training_record(r) for r in train if r["label"] != C.NOISE_LABEL]
    val_kept = [_to_training_record(r) for r in val if r["label"] != C.NOISE_LABEL]
    # test 는 4-class 그대로 유지 (label_id 는 OUTPUT 공간).

    n_train_dropped = len(train) - len(train_kept)
    n_val_dropped = len(val) - len(val_kept)
    n_test_noise = sum(1 for r in test if r["label"] == C.NOISE_LABEL)
    print(
        f"\n[3-class training] dropped NOISE: train -{n_train_dropped}, "
        f"val -{n_val_dropped}; test 에 NOISE {n_test_noise} 건 유지 (rejection 평가용)"
    )

    show_stats("train", train_kept)
    show_stats("val", val_kept)
    show_stats("test", test)

    write_jsonl(C.DATA_DIR / "train.jsonl", train_kept)
    write_jsonl(C.DATA_DIR / "val.jsonl", val_kept)
    write_jsonl(C.DATA_DIR / "test.jsonl", test)

    # class weights — 3-class training space
    counts = Counter(r["label_id"] for r in train_kept)
    total = max(len(train_kept), 1)
    weights = [total / (C.NUM_LABELS * max(counts.get(i, 0), 1))
               for i in range(C.NUM_LABELS)]
    (C.DATA_DIR / "class_weights.json").write_text(
        json.dumps(weights, ensure_ascii=False), encoding="utf-8"
    )
    print("\nclass weights (train, 3-class):")
    for i, w in enumerate(weights):
        print(f"  {C.TRAINING_ID2LABEL[i]:18s} n={counts.get(i, 0):5d}  w={w:.3f}")
    print(f"\nartifacts → {C.DATA_DIR}")


if __name__ == "__main__":
    main()
