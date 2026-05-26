"""NOISE 라벨 추가 마이닝 — GPT-4.1 teacher.

4-class 학습에서 NOISE F1=0.60 정체 → NOISE class support 증가용 데이터 보강.

두 가지 모드:

  1. synthetic  : GPT-4.1 로 다양한 카테고리의 NOISE 댓글 생성 (입력 불필요)
                  - 단순 반응 / 밈 / 음악 질문 / 일상 잡담 / 광고 / 다른 주제 등

  2. label      : 사용자가 제공한 raw 댓글 JSONL 을 GPT-4.1 로 라벨링 후
                  NOISE 만 추출

출력 형식은 ``comment_labels/labeled_gpt41_azure.jsonl`` 과 호환되며,
별도 파일(`labeled_gpt41_azure_noise_extra.jsonl`)에 append 한다.
검토 후 본 라벨 파일에 cat 으로 합치면 `prepare_dataset` 가 자동 인식.

사용 예:
    python -m local_classifier.mine_noise --mode synthetic --count 500
    python -m local_classifier.mine_noise --mode label --input raw.jsonl

필수 환경변수 (RunYourAI 게이트웨이 — OpenAI 호환 endpoint):
    RUNYOURAI_API_KEY        (필수)
    RUNYOURAI_BASE_URL       (기본 https://api.runyour.ai/v1)
    RUNYOURAI_MODEL          (기본 openai/gpt-4.1-2025-04-14)

또는 표준 OpenAI 사용 시:
    OPENAI_API_KEY           (필수)
    OPENAI_BASE_URL          (선택)
    OPENAI_MODEL             (기본 gpt-4o)

추가 의존성: pip install langchain-openai

비용 추정 (GPT-4.1 ~$2 in / $8 out per 1M tokens):
    synthetic 500건  ≈  $0.30
    label 5000건    ≈  $0.60
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from local_classifier import config as C


REPO_ROOT = C.REPO_ROOT
DEFAULT_OUTPUT = REPO_ROOT / "comment_labels" / "labeled_gpt41_azure_noise_extra.jsonl"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYNTHETIC_SYSTEM_PROMPT = """당신은 한국 유튜브 테크 리뷰 영상의 댓글 데이터를 생성하는 전문 어노테이터입니다.
주어진 카테고리에 정확히 부합하는 **NOISE** 댓글을 다양한 길이·어조·패턴으로 생성합니다.

NOISE 정의: 제품 평가도, 영상/리뷰어 반응도, 제품 관련 질문도 **아닌** 댓글.
구체 카테고리: 단순 반응(ㅋㅋ/와/대박), 밈/유행어, 음악·BGM 질문, 다른 주제 잡담, 일상 안부,
광고 의심, 욕설/저품질, 영상에 등장한 사람의 외모/사적 언급 등.

반드시 JSON 배열로만 응답. 각 원소는 {"text": "..."} 형식. 다른 키 금지."""


LABEL_SYSTEM_PROMPT = """당신은 한국 유튜브 테크 리뷰 영상의 댓글을 4-class로 분류하는 전문 어노테이터입니다.

라벨:
- PRODUCT_OPINION: 제품 성능/품질/가격/디자인 등 제품 자체에 대한 평가
- VIDEO_REACTION : 영상/리뷰어/편집/연출에 대한 반응 (제품 외)
- QUESTION       : 제품 관련 질문 (영상 자체 질문 아님)
- NOISE          : 위 3개에 모두 해당 안 됨 (단순 반응/밈/음악·BGM 질문/일상 잡담/광고/다른 주제)

각 댓글에 대해 라벨 + confidence(0.0~1.0) + 짧은 이유를 출력.
반드시 JSON 배열로만 응답. 각 원소:
{"i": <index>, "label": "...", "confidence": 0.95, "reason": "..."}
다른 키나 텍스트 금지."""


SYNTHETIC_CATEGORIES = [
    ("단순 반응/감탄사",            "ㅋㅋㅋ, 와, 헐, 대박, 미쳤다 같은 짧은 감탄. 의미 정보 없음."),
    ("밈/유행어",                  "최근 한국 인터넷 밈, 유행어, 드립. 제품·영상과 무관."),
    ("음악·BGM·썸네일 질문",        "배경음악 제목, BGM, 썸네일 디자인 문의."),
    ("영상 출연자 외모/사적 언급",   "리뷰어 목소리/외모/머리 등 사적 코멘트. 제품·영상 내용 무관."),
    ("일상 안부/잡담",              "오늘 날씨, 점심 메뉴, 주말 인사 등 비관련 잡담."),
    ("광고 의심/스팸",              "단축 URL, 광고성 멘트, 다른 채널 홍보."),
    ("욕설/저품질",                "단순 욕설이나 의미 없는 키보드 입력."),
    ("다른 주제 (영상 무관)",       "정치·스포츠·연예 등 영상과 무관한 화제."),
]


# ---------------------------------------------------------------------------
# Heuristic candidate filter (옵션) — 라벨 호출 비용 절감
# ---------------------------------------------------------------------------

NOISE_KEYWORDS = re.compile(
    r"(배경음악|브금|bgm|BGM|썸네일|편집|음악|광고|홍보|"
    r"날씨|밥|점심|저녁|아침|머리|얼굴|목소리|성우|"
    r"ㅋ{3,}|ㅎ{3,}|ㅠ{3,}|ㅜ{3,})"
)
SHORT_THRESHOLD = 5  # 5자 이하면 NOISE 후보


def looks_like_noise(text: str) -> bool:
    """경량 휴리스틱 — GPT-4.1 라벨링 전 후보 필터링."""
    t = text.strip()
    if len(t) <= SHORT_THRESHOLD:
        return True
    if NOISE_KEYWORDS.search(t):
        return True
    # 같은 글자 반복 (긴 호응형)
    if re.search(r"(.)\1{4,}", t):
        return True
    return False


# ---------------------------------------------------------------------------
# LLM 호출
# ---------------------------------------------------------------------------

def _get_llm(temperature: float):
    """ChatOpenAI 인스턴스 — RunYourAI(OpenAI 호환) 또는 표준 OpenAI 지원.

    환경변수 우선순위:
      1. RUNYOURAI_API_KEY  → RunYourAI 게이트웨이 사용 (Moabom 운영 표준)
      2. OPENAI_API_KEY     → 표준 OpenAI 사용
    둘 다 없으면 RuntimeError.

    `scripts.llm.get_chat_llm` 이 import 가능하면 그쪽을 우선 사용 (Moabom_Prototype
    내부에서 호출 시). LocalModelTraining 같은 standalone 리포에서는 import 실패
    하므로 직접 ChatOpenAI 생성.
    """
    import os

    # 1) Moabom_Prototype 내부 호출이면 표준 진입점 우선 시도
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.llm import get_chat_llm  # type: ignore
        return get_chat_llm(temperature=temperature, max_tokens=4000)
    except (ImportError, ModuleNotFoundError):
        pass  # standalone 리포 — 아래에서 직접 처리

    # 2) standalone: 환경변수로 직접 ChatOpenAI 생성
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise RuntimeError(
            "langchain-openai 미설치. 다음 실행:\n"
            "  pip install langchain-openai"
        )

    runyour_key = os.environ.get("RUNYOURAI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if runyour_key:
        return ChatOpenAI(
            api_key=runyour_key,
            base_url=os.environ.get("RUNYOURAI_BASE_URL", "https://api.runyour.ai/v1"),
            model=os.environ.get("RUNYOURAI_MODEL", "openai/gpt-4.1-2025-04-14"),
            temperature=temperature,
            max_tokens=4000,
        )
    if openai_key:
        kwargs: dict = {
            "api_key": openai_key,
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o"),
            "temperature": temperature,
            "max_tokens": 4000,
        }
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)

    raise RuntimeError(
        "API key 환경변수 미설정. RUNYOURAI_API_KEY 또는 OPENAI_API_KEY 중 하나 필요.\n"
        "  export RUNYOURAI_API_KEY=...     # RunYourAI (권장)\n"
        "  export OPENAI_API_KEY=...        # 또는 표준 OpenAI"
    )


def _call_llm_json(llm, system: str, user: str, retries: int = 3) -> list[dict]:
    """LLM 호출 후 JSON 배열만 파싱. 실패 시 retry."""
    from langchain_core.messages import HumanMessage, SystemMessage

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
            text = resp.content.strip()
            # ```json fence 제거
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
            arr = json.loads(text)
            if not isinstance(arr, list):
                raise ValueError(f"응답이 JSON 배열 아님: {type(arr)}")
            return arr
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  [retry {attempt+1}/{retries}] {e}")
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM JSON 파싱 {retries}회 실패: {last_err}")


# ---------------------------------------------------------------------------
# Synthetic mode
# ---------------------------------------------------------------------------

def generate_synthetic(count: int, batch_id: str) -> list[dict]:
    """카테고리를 균등 분배해 NOISE 댓글 생성."""
    llm = _get_llm(temperature=0.9)  # 다양성 위해 temperature 높임
    per_cat = max(count // len(SYNTHETIC_CATEGORIES), 5)
    out: list[dict] = []
    for cat_name, cat_desc in SYNTHETIC_CATEGORIES:
        if len(out) >= count:
            break
        user = (
            f"카테고리: {cat_name}\n"
            f"설명: {cat_desc}\n\n"
            f"위 카테고리에 정확히 부합하는 **NOISE** 댓글을 정확히 {per_cat}개 생성하세요.\n"
            f"다양한 길이·어조·맞춤법 변형을 포함. 실제 유튜브 댓글 같은 자연스러움 유지.\n"
            f"JSON 배열만 출력: [{{\"text\": \"...\"}}, ...]"
        )
        print(f"  [{cat_name}] 요청 중...")
        try:
            arr = _call_llm_json(llm, SYNTHETIC_SYSTEM_PROMPT, user)
        except Exception as e:
            print(f"  [{cat_name}] FAIL: {e}")
            continue
        for i, rec in enumerate(arr):
            text = (rec.get("text") or "").strip()
            if not text or len(text) < C.MIN_TEXT_LEN or len(text) > C.MAX_TEXT_LEN:
                continue
            out.append(_make_record(
                text=text,
                comment_id=f"synth-{batch_id}-{cat_name[:4]}-{i:03d}",
                video_id=f"synthetic-{batch_id}-{cat_name[:4]}",
                confidence=0.95,
                reasoning=f"synthetic NOISE ({cat_name})",
            ))
        print(f"  [{cat_name}] 누적 {len(out)}/{count}")
        if len(out) >= count:
            out = out[:count]
            break
    return out


# ---------------------------------------------------------------------------
# Label mode — 외부 raw 댓글 라벨링 후 NOISE 만 추출
# ---------------------------------------------------------------------------

def label_external(
    input_path: Path,
    batch_size: int = 25,
    apply_heuristic: bool = True,
    text_field: str = "text",
    video_field: str = "video_id",
) -> list[dict]:
    """raw 댓글 JSONL 을 GPT-4.1 로 라벨링 → NOISE 만 반환."""
    raw: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = (rec.get(text_field) or "").strip()
            if not text or len(text) < C.MIN_TEXT_LEN or len(text) > C.MAX_TEXT_LEN:
                continue
            raw.append({"text": text, "video_id": rec.get(video_field) or "unknown",
                        "comment_id": rec.get("comment_id") or str(uuid.uuid4())[:12]})
    print(f"loaded raw comments: {len(raw)}")

    if apply_heuristic:
        before = len(raw)
        raw = [r for r in raw if looks_like_noise(r["text"])]
        print(f"heuristic pre-filter: {before} -> {len(raw)} candidates "
              f"(short ≤{SHORT_THRESHOLD} OR keyword OR repeated chars)")

    if not raw:
        return []

    llm = _get_llm(temperature=0.0)
    out: list[dict] = []
    for start in range(0, len(raw), batch_size):
        batch = raw[start:start + batch_size]
        items = "\n".join(f"  {i}. {r['text']}" for i, r in enumerate(batch))
        user = f"분류할 댓글 {len(batch)}개:\n{items}\n\nJSON 배열로 출력."
        print(f"  batch {start//batch_size + 1}/{(len(raw)-1)//batch_size + 1} ({len(batch)}건)")
        try:
            arr = _call_llm_json(llm, LABEL_SYSTEM_PROMPT, user)
        except Exception as e:
            print(f"  batch FAIL: {e}")
            continue
        for entry in arr:
            try:
                idx = int(entry["i"])
                label = entry["label"]
                conf = float(entry["confidence"])
            except (KeyError, TypeError, ValueError):
                continue
            if label != "NOISE":
                continue
            if conf < C.MIN_CONFIDENCE:
                continue
            if not (0 <= idx < len(batch)):
                continue
            src = batch[idx]
            out.append(_make_record(
                text=src["text"],
                comment_id=src["comment_id"],
                video_id=src["video_id"],
                confidence=conf,
                reasoning=entry.get("reason", "")[:200],
            ))
        print(f"  누적 NOISE 확정: {len(out)}")
    return out


# ---------------------------------------------------------------------------
# Record format (labeled_gpt41_azure.jsonl 호환)
# ---------------------------------------------------------------------------

def _make_record(
    *,
    text: str,
    comment_id: str,
    video_id: str,
    confidence: float,
    reasoning: str,
) -> dict:
    return {
        "comment_id": comment_id,
        "video_id": video_id,
        "product_id": None,
        "text": text,
        "label": "NOISE",
        "confidence": round(confidence, 4),
        "label_scores": {"NOISE": round(confidence, 4)},
        "teacher_model": "openai/gpt-4.1-2025-04-14",
        "final_action": "EXCLUDE",
        "exclusion_reason": "NOISE",
        "is_product_related": False,
        "like_count": 0,
        "reply_count": 0,
        "classified_at": datetime.now(timezone.utc).isoformat(),
        "reasoning": reasoning,
    }


def write_jsonl(path: Path, records: list[dict], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["synthetic", "label"], required=True)
    ap.add_argument("--count", type=int, default=500,
                    help="(synthetic) 생성 댓글 수")
    ap.add_argument("--input", type=str, default=None,
                    help="(label) raw 댓글 JSONL 경로 (text 필드 필수)")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--video-field", default="video_id")
    ap.add_argument("--no-heuristic", action="store_true",
                    help="(label) 휴리스틱 pre-filter 건너뛰기 — 전수 라벨링")
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    ap.add_argument("--append", action="store_true",
                    help="기존 출력 파일에 append (기본은 덮어쓰기)")
    args = ap.parse_args()

    output_path = Path(args.output)
    if args.mode == "synthetic":
        batch_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f"[mode=synthetic] count={args.count} batch_id={batch_id}")
        recs = generate_synthetic(count=args.count, batch_id=batch_id)
    else:  # label
        if not args.input:
            ap.error("--mode label 사용 시 --input 필수")
        print(f"[mode=label] input={args.input} heuristic={not args.no_heuristic}")
        recs = label_external(
            input_path=Path(args.input),
            batch_size=args.batch_size,
            apply_heuristic=not args.no_heuristic,
            text_field=args.text_field,
            video_field=args.video_field,
        )

    if not recs:
        print("생성된 레코드 0건. 출력 파일 갱신 안 함.")
        return

    write_jsonl(output_path, recs, append=args.append)
    print(f"\n총 {len(recs)}건 NOISE → {output_path} ({'append' if args.append else 'overwrite'})")
    print()
    print("학습 데이터에 합치려면:")
    print(f"  cat {output_path} >> {REPO_ROOT}/comment_labels/labeled_gpt41_azure.jsonl")
    print(f"  python -m local_classifier.prepare_dataset")
    print(f"  python -m local_classifier.train")


if __name__ == "__main__":
    main()
