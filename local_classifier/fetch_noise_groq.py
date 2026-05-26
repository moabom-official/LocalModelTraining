"""YouTube API 로 댓글 fetch + Groq Llama 로 NOISE 필터링.

mine_noise.py (GPT-4.1) 대비 cheap/fast 대안. Llama 3.3 70B via Groq.

파이프라인:
  1. video_ids 확보 (CLI / 파일 / 기존 labeled 데이터)
  2. YouTube Data API v3 로 top-level 댓글 fetch
  3. 휴리스틱 pre-filter (짧음 / 음악·광고 키워드 / 반복문자) — NOISE 후보로 좁힘
  4. Groq Llama 3.3 70B batch 분류
  5. NOISE + confidence >= 0.85 만 추출 (target 건수까지)
  6. labeled_gpt41_azure.jsonl 호환 JSONL 출력

비용 / 속도 (대략):
  YouTube API   : 무료 (할당량 10K units/day, comment fetch = 1 unit)
  Groq Llama 70B: ~$0.0006 in / $0.0008 out per 1K tokens
  1000 건 분류  : ~$0.05, <2분

환경변수 (필수):
  YOUTUBE_API_KEY        YouTube Data API v3 key
  GROQ_API_KEY           Groq key (https://console.groq.com)
  GROQ_MODEL             (기본 llama-3.3-70b-versatile)

설치:
  pip install google-api-python-client groq

사용:
  # 기본: 기존 labeled 데이터의 video_id 사용 → 20 영상 × 50 댓글 = 1000건 → NOISE 200건
  python -m local_classifier.fetch_noise_groq

  # 특정 영상
  python -m local_classifier.fetch_noise_groq --video-ids xxx,yyy --per-video 100

  # 영상 ID 파일 (한 줄에 하나)
  python -m local_classifier.fetch_noise_groq --video-ids-file ids.txt --target 500
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from local_classifier import config as C
from local_classifier.mine_noise import looks_like_noise, write_jsonl

REPO_ROOT = C.REPO_ROOT
DEFAULT_OUTPUT = REPO_ROOT / "comment_labels" / "labeled_groq_noise_extra.jsonl"

GROQ_SYSTEM_PROMPT = """당신은 한국 유튜브 테크 리뷰 댓글을 4-class 로 분류하는 어노테이터입니다.

라벨:
- PRODUCT_OPINION: 제품 자체에 대한 평가 (성능/배터리/가격/디자인 등)
- VIDEO_REACTION : 영상/리뷰어/편집/연출에 대한 반응
- QUESTION       : 제품 관련 질문
- NOISE          : 위 셋에 모두 해당 안 됨 (단순 반응/밈/음악·BGM/일상 잡담/광고/다른 주제)

JSON 배열로만 응답. 각 원소:
{"i": <index>, "label": "...", "confidence": 0.0-1.0}
다른 텍스트 절대 금지."""


# ---------------------------------------------------------------------------
# Video ID 수집
# ---------------------------------------------------------------------------

def get_video_ids_from_labeled() -> list[str]:
    """기존 labeled_gpt41_azure.jsonl 에서 unique video_id 추출 (synthetic 제외)."""
    path = REPO_ROOT / "comment_labels" / "labeled_gpt41_azure.jsonl"
    if not path.exists():
        return []
    vids: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = rec.get("video_id")
            if vid and not str(vid).startswith("synthetic"):
                vids.add(str(vid))
    return sorted(vids)


# ---------------------------------------------------------------------------
# YouTube fetch
# ---------------------------------------------------------------------------

def fetch_youtube_comments(video_ids: list[str], per_video: int = 50) -> list[dict]:
    """Top-level 댓글만. order='time' 으로 최신 mix 확보."""
    try:
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
    except ImportError:
        raise RuntimeError(
            "google-api-python-client 미설치.  pip install google-api-python-client"
        )

    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY 환경변수 필수.")

    yt = build("youtube", "v3", developerKey=api_key, cache_discovery=False)
    out: list[dict] = []
    for vid in video_ids:
        fetched = 0
        next_page = None
        while fetched < per_video:
            try:
                resp = yt.commentThreads().list(
                    part="snippet",
                    videoId=vid,
                    maxResults=min(100, per_video - fetched),
                    pageToken=next_page,
                    textFormat="plainText",
                    order="time",
                ).execute()
            except HttpError as e:
                # 댓글 비활성 / 영상 삭제 / 권한 등 — 건너뜀
                print(f"  [{vid}] skip: {e.resp.status if hasattr(e, 'resp') else e}")
                break
            for item in resp.get("items", []):
                snip = item["snippet"]["topLevelComment"]["snippet"]
                text = (snip.get("textDisplay") or "").strip()
                if not text:
                    continue
                out.append({
                    "comment_id": item["id"],
                    "video_id": vid,
                    "text": text,
                    "like_count": snip.get("likeCount", 0),
                    "reply_count": item["snippet"].get("totalReplyCount", 0),
                })
                fetched += 1
            next_page = resp.get("nextPageToken")
            if not next_page:
                break
        print(f"  [{vid}] fetched: {fetched}")
    return out


# ---------------------------------------------------------------------------
# Groq classify
# ---------------------------------------------------------------------------

def classify_with_groq(
    comments: list[dict],
    batch_size: int = 25,
    retries: int = 3,
) -> list[tuple[dict, str, float]]:
    """(comment_dict, label, confidence) 리스트 반환. 파싱 실패한 항목은 drop."""
    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError(
            "groq 미설치.  pip install groq"
        )

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY 환경변수 필수. https://console.groq.com")

    client = Groq(api_key=api_key)
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    results: list[tuple[dict, str, float]] = []
    n_batches = (len(comments) + batch_size - 1) // batch_size
    for bi, start in enumerate(range(0, len(comments), batch_size), 1):
        batch = comments[start:start + batch_size]
        items = "\n".join(f"{i}. {c['text']}" for i, c in enumerate(batch))
        prompt = f"분류할 댓글 {len(batch)}개:\n{items}\n\nJSON 배열만 출력."

        arr: list | None = None
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": GROQ_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=2000,
                )
                text = resp.choices[0].message.content.strip()
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
                arr = json.loads(text)
                if not isinstance(arr, list):
                    raise ValueError(f"응답이 배열 아님: {type(arr)}")
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
        if arr is None:
            print(f"  batch {bi}/{n_batches} FAIL ({retries}회): {last_err}")
            continue

        for entry in arr:
            try:
                idx = int(entry["i"])
                label = entry["label"]
                conf = float(entry["confidence"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (0 <= idx < len(batch)):
                continue
            results.append((batch[idx], label, conf))
        print(f"  batch {bi}/{n_batches} ok ({len(arr)} parsed, {len(results)} cum)")
    return results


# ---------------------------------------------------------------------------
# Output record
# ---------------------------------------------------------------------------

def make_record(src: dict, confidence: float, model_name: str) -> dict:
    return {
        "comment_id": src["comment_id"],
        "video_id": src["video_id"],
        "product_id": None,
        "text": src["text"],
        "label": "NOISE",
        "confidence": round(confidence, 4),
        "label_scores": {"NOISE": round(confidence, 4)},
        "teacher_model": model_name,
        "final_action": "EXCLUDE",
        "exclusion_reason": "NOISE",
        "is_product_related": False,
        "like_count": src.get("like_count", 0),
        "reply_count": src.get("reply_count", 0),
        "classified_at": datetime.now(timezone.utc).isoformat(),
        "reasoning": f"Groq classifier ({model_name}); conf {confidence:.2f}",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="YouTube + Groq NOISE 마이닝 (mine_noise.py 의 cheap 대안)"
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--video-ids", help="콤마로 구분된 video ID 리스트")
    src.add_argument("--video-ids-file", help="한 줄에 하나씩 video ID 가 있는 파일")
    ap.add_argument("--per-video", type=int, default=50, help="영상 당 fetch 댓글 수")
    ap.add_argument("--max-videos", type=int, default=20, help="처리할 영상 최대 개수")
    ap.add_argument("--target", type=int, default=200, help="목표 NOISE 라벨 수")
    ap.add_argument("--no-heuristic", action="store_true",
                    help="휴리스틱 pre-filter 건너뛰기 (전수 Groq)")
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    # 1) video IDs 확보
    if args.video_ids:
        video_ids = [v.strip() for v in args.video_ids.split(",") if v.strip()]
        src_desc = "CLI --video-ids"
    elif args.video_ids_file:
        with open(args.video_ids_file, encoding="utf-8") as f:
            video_ids = [line.strip() for line in f if line.strip()]
        src_desc = f"file {args.video_ids_file}"
    else:
        video_ids = get_video_ids_from_labeled()
        src_desc = "labeled_gpt41_azure.jsonl"

    if not video_ids:
        print("[ERROR] video ID 0건. --video-ids / --video-ids-file 로 직접 제공하세요.")
        sys.exit(1)

    random.Random(42).shuffle(video_ids)
    video_ids = video_ids[:args.max_videos]
    print(f"source: {src_desc}")
    print(f"video IDs: {len(video_ids)} (max {args.max_videos})")

    # 2) fetch
    print(f"\n[1] YouTube fetch (~{args.per_video}/video)")
    raw = fetch_youtube_comments(video_ids, per_video=args.per_video)
    print(f"  total fetched: {len(raw)}")

    # 길이 정제
    raw = [r for r in raw if C.MIN_TEXT_LEN <= len(r["text"]) <= C.MAX_TEXT_LEN]
    print(f"  after length filter: {len(raw)}")
    if not raw:
        print("[ERROR] 가공 후 0건.")
        sys.exit(1)

    # 중복 제거 (텍스트 기준 — 같은 영상 다른 사용자가 같은 댓글)
    seen: set[str] = set()
    dedup: list[dict] = []
    for r in raw:
        key = r["text"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    print(f"  after dedup: {len(dedup)}")

    # 3) heuristic
    if args.no_heuristic:
        candidates = dedup
        print(f"\n[2] heuristic skipped: all {len(candidates)} -> Groq")
    else:
        candidates = [r for r in dedup if looks_like_noise(r["text"])]
        print(f"\n[2] heuristic pre-filter: {len(dedup)} -> {len(candidates)} candidates")

    if not candidates:
        print("[ERROR] 후보 0건. --no-heuristic 시도해보세요.")
        sys.exit(1)

    # 4) Groq classify
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    print(f"\n[3] Groq classification (batch {args.batch_size}, model {model})")
    classified = classify_with_groq(candidates, batch_size=args.batch_size)
    print(f"  total classified: {len(classified)}")

    # 5) NOISE + conf 필터
    label_counts: dict = {}
    confirmed: list[dict] = []
    for src, label, conf in classified:
        label_counts[label] = label_counts.get(label, 0) + 1
        if label != "NOISE":
            continue
        if conf < C.MIN_CONFIDENCE:
            continue
        confirmed.append(make_record(src, conf, model))
        if len(confirmed) >= args.target:
            break

    print(f"\n[4] label distribution: {label_counts}")
    print(f"  NOISE confirmed (conf >= {C.MIN_CONFIDENCE}): "
          f"{len(confirmed)} / target {args.target}")

    if not confirmed:
        print("[WARN] confirmed NOISE 0. 출력 파일 갱신 안 함.")
        return

    out_path = Path(args.output)
    write_jsonl(out_path, confirmed, append=args.append)
    print(f"\n저장 완료: {out_path} ({len(confirmed)} 건, "
          f"{'append' if args.append else 'overwrite'})")
    print()
    print("학습 데이터 합치기:")
    print(f"  cat {out_path} >> {REPO_ROOT}/comment_labels/labeled_gpt41_azure.jsonl")
    print(f"  python -m local_classifier.prepare_dataset")
    print(f"  python -m local_classifier.train")


if __name__ == "__main__":
    main()
