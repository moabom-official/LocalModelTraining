"""artifacts/<model_slug>/logs/test_summary.json 들을 읽어 모델별 비교 표 출력.

각 모델별로 학습/평가를 마치면 자동으로 결과가 ``artifacts/<slug>/logs/``
에 떨어진다 (config.py 가 MODEL_SLUG 기반으로 디렉토리 분리).
이 스크립트는 그 결과들을 한 화면에서 비교한다.

Usage:
    python -m local_classifier.compare_models

    # 특정 디렉토리만:
    python -m local_classifier.compare_models --output /data/moabom_artifacts

Output:
    공통 4 클래스 confusion + per-class F1 + macro F1 + accuracy 표.
    모델별 best epoch (training_config.json 에서) 도 함께 출력.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_classifier import config as C


def per_class_f1(cm: list[list[int]]) -> tuple[list[float], float]:
    n = len(cm)
    f1s = []
    for c in range(n):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(n) if r != c)
        fn = sum(cm[c][r] for r in range(n) if r != c)
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        f1s.append(f1)
    return f1s, sum(f1s) / n


def gather(root: Path) -> list[dict]:
    results = []
    if not root.exists():
        return results
    for slug_dir in sorted(root.iterdir()):
        if not slug_dir.is_dir():
            continue
        summary_path = slug_dir / "logs" / "test_summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[skip] {slug_dir.name}: {e}")
            continue

        # training_config 추가 정보
        train_cfg_path = slug_dir / "model" / "best" / "training_config.json"
        train_cfg = {}
        if train_cfg_path.exists():
            try:
                train_cfg = json.loads(train_cfg_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        cm = summary.get("cm", [])
        f1s, macro = per_class_f1(cm) if cm else ([], summary.get("macro_f1", 0.0))
        results.append({
            "slug": slug_dir.name,
            "model": train_cfg.get("base_model") or slug_dir.name.replace("__", "/"),
            "acc": summary.get("acc", 0.0),
            "macro_f1": summary.get("macro_f1", macro),
            "per_class_f1": f1s,
            "n": summary.get("n", 0),
            "best_epoch": train_cfg.get("epoch"),
            "best_val_macro_f1": train_cfg.get("best_val_macro_f1"),
            "labels": summary.get("labels", C.LABEL_NAMES),
        })
    return results


def fmt_pcf(f1s: list[float], labels: list[str]) -> str:
    if not f1s:
        return "n/a"
    return "  ".join(f"{lbl[:3]}={f:.3f}" for lbl, f in zip(labels, f1s))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", type=str, default=str(C.OUTPUT_DIR),
                    help=f"artifacts 루트 (default: {C.OUTPUT_DIR})")
    args = ap.parse_args()

    root = Path(args.output)
    results = gather(root)

    if not results:
        print(f"No test_summary.json found under {root}/*/logs/")
        print("학습 → 평가를 먼저 실행하세요:")
        print("  python -m local_classifier.train")
        print("  python -m local_classifier.evaluate")
        return

    # 정렬: macro_f1 내림차순
    results.sort(key=lambda x: x["macro_f1"], reverse=True)

    # 라벨은 첫 결과 기준 (모두 동일해야 함)
    labels = results[0]["labels"]
    n_labels = len(labels)

    print(f"\n{'=' * 100}")
    print(f"모델 비교 (data: {root})")
    print(f"{'=' * 100}")
    print(f"{'model':45s} {'n':>4s} {'acc':>6s} {'macroF1':>8s} {'best_ep':>7s} {'best_valF1':>11s}")
    print("-" * 100)
    for r in results:
        ep = str(r["best_epoch"]) if r["best_epoch"] is not None else "-"
        vf1 = f"{r['best_val_macro_f1']:.4f}" if r["best_val_macro_f1"] else "-"
        print(f"{r['model'][:45]:45s} {r['n']:>4d} {r['acc']:>6.3f} "
              f"{r['macro_f1']:>8.4f} {ep:>7s} {vf1:>11s}")

    # per-class F1 상세
    print()
    print(f"{'=' * 100}")
    print("Per-class F1")
    print(f"{'=' * 100}")
    header = f"{'model':45s}  " + "  ".join(f"{lbl[:8]:>8s}" for lbl in labels)
    print(header)
    print("-" * 100)
    for r in results:
        pcf = "  ".join(f"{f:>8.3f}" for f in r["per_class_f1"])
        print(f"{r['model'][:45]:45s}  {pcf}")

    # 1등 vs 2등 격차
    if len(results) >= 2:
        first, second = results[0], results[1]
        delta = first["macro_f1"] - second["macro_f1"]
        print()
        print(f"=> winner: {first['model']}  (macro F1 +{delta:.4f} vs {second['model']})")


if __name__ == "__main__":
    main()
