"""클래스별 라벨 데이터 마이닝 — GPT-4.1 teacher (via RunYourAI 게이트웨이).

4-class 학습의 약한 클래스(NOISE / VIDEO_REACTION) support 증가용 도구.

세 가지 모드:

  1. synthetic  : GPT-4.1 로 카테고리별 합성 댓글 생성 (입력 불필요)
                  NOISE 권장. VR 은 stereotype 위험 → youtube 모드 권장.

  2. label      : 사용자가 제공한 raw 댓글 JSONL 을 GPT-4.1 로 분류 후
                  --label 지정 라벨만 추출 (NOISE / VR 등 자유)

  3. youtube    : ★ 실제 YouTube 댓글 fetch + GPT-4.1 분류 + 라벨 추출 ★
                  합성 데이터의 stereotype 한계를 우회.
                  video_ids 미지정 시 labeled_gpt41_azure.jsonl 에서 자동 추출.

출력 형식은 ``comment_labels/labeled_gpt41_azure.jsonl`` 과 호환되며,
별도 파일(`labeled_gpt41_azure_<label>_extra.jsonl`)에 append 한다.
검토 후 본 라벨 파일에 cat 으로 합치면 `prepare_dataset` 가 자동 인식.

사용 예:
    # NOISE 합성
    python -m local_classifier.mine_noise --mode synthetic --label NOISE --count 500

    # ★ VR 실 댓글 마이닝 (권장)
    python -m local_classifier.mine_noise --mode youtube --label VIDEO_REACTION \\
        --max-videos 50 --per-video 100 --target 300

    # ★ NOISE 실 댓글 마이닝
    python -m local_classifier.mine_noise --mode youtube --label NOISE \\
        --max-videos 30 --target 300

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
import random
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

def _system_prompt(label: str, definition: str) -> str:
    return (
        f"당신은 한국 유튜브 테크 리뷰 영상의 댓글 데이터를 생성하는 전문 어노테이터입니다.\n"
        f"주어진 카테고리에 정확히 부합하는 **{label}** 댓글을 다양한 길이·어조·패턴으로 생성합니다.\n\n"
        f"{label} 정의: {definition}\n\n"
        f'반드시 JSON 배열로만 응답. 각 원소는 {{"text": "..."}} 형식. 다른 키 금지.'
    )


CLASSIFY_SYSTEM_PROMPT = """당신은 한국 유튜브 테크 리뷰 영상의 댓글을 4-class 로 분류하는 어노테이터입니다.

라벨:
- PRODUCT_OPINION: 제품 자체(성능/배터리/가격/디자인/화면 등)에 대한 평가
- VIDEO_REACTION : 영상·리뷰어·편집·자막·촬영·톤·구독 등 영상 제작물에 대한 반응
- QUESTION       : 제품 관련 질문 (영상 자체 질문 아님)
- NOISE          : 위 3개 모두 해당 안 됨 (단순반응·밈·음악질문·잡담·광고·욕설·외모언급 등)

각 댓글에 대해 라벨 + confidence(0.0~1.0) + 짧은 이유 출력.
반드시 JSON 배열로만 응답. 각 원소:
{"i": <index>, "label": "...", "confidence": 0.95, "reason": "..."}
다른 텍스트 금지."""


# 라벨별 정의 + 카테고리 (label-aware 합성)
LABEL_DEFINITIONS: dict[str, str] = {
    "NOISE": (
        "제품 평가도, 영상/리뷰어 반응도, 제품 관련 질문도 **아닌** 댓글. "
        "구체 카테고리: 단순 반응(ㅋㅋ/와/대박), 밈/유행어, 음악·BGM 질문, "
        "다른 주제 잡담, 일상 안부, 광고 의심, 욕설/저품질, "
        "영상에 등장한 사람의 외모/사적 언급 등."
    ),
    "VIDEO_REACTION": (
        "영상 자체, 리뷰어, 편집, 연출, 자막, 카메라워크, 영상 길이, 톤·발성 등 "
        "**제품 외** 영상 제작물 자체에 대한 반응. 제품 특성(성능/배터리/가격/디자인) 평가는 "
        "PRODUCT_OPINION 이므로 절대 포함 금지."
    ),
    "QUESTION": (
        "제품에 대한 질문 댓글. 성능/배터리/가격/구매처/호환성/비교/사용법 등 "
        "**제품 관련** 정보를 묻는 의문문. 영상 자체나 리뷰어에 대한 질문(예: 다음 영상 언제?)"
        "은 VIDEO_REACTION 으로 분류되므로 제외."
    ),
}

SYNTHETIC_CATEGORIES: dict[str, list[tuple[str, str]]] = {
    "NOISE": [
        # "단순 반응/감탄사" 는 짧아서 prepare_dataset dedup 에 거의 다 잘림 → 제거.
        # 짧은 NOISE 는 운영 라벨에 이미 충분 (CHATTER 405 의 대다수).
        ("밈/유행어 (긴 문장)",        "최근 한국 인터넷 밈, 유행어, 드립을 활용한 한 문장 이상의 댓글. 제품·영상 무관."),
        ("음악·BGM·썸네일 질문",        "배경음악 제목, BGM, 썸네일 디자인 문의."),
        ("영상 출연자 외모/사적 언급",   "리뷰어 목소리/외모/머리 등 사적 코멘트. 제품·영상 내용 무관."),
        ("일상 안부/잡담",              "오늘 날씨, 점심 메뉴, 주말 인사 등 비관련 잡담."),
        ("광고 의심/스팸",              "단축 URL, 광고성 멘트, 다른 채널 홍보."),
        ("욕설/저품질",                "단순 욕설이나 의미 없는 키보드 입력."),
        ("다른 주제 (영상 무관)",       "정치·스포츠·연예 등 영상과 무관한 화제."),
    ],
    "VIDEO_REACTION": [
        ("영상 자체 칭찬",              "영상 잘 만들었네요/재밌어요/퀄리티 좋네요 류. 제품 언급 없이 영상 자체에 대한 호평."),
        ("리뷰어 설명력 평가",          "설명 잘 하시네요/이해 쏙쏙됨/쉽게 풀어주셔서 좋아요 등 화법·교수력 평가."),
        ("편집·연출 칭찬·비판",         "편집 깔끔하다/컷 좋네요/효과 과하다 등 편집·연출 자체에 대한 평가."),
        ("자막·디자인 코멘트",          "자막 가독성/색감/오타 지적/자막 잘 달아주세요 등 자막·자료 화면 코멘트."),
        ("톤·발성·발음",                "목소리 좋네요/발음 명확/말 빠르네요 등 리뷰어 음성 평가 (외모 사적 언급은 NOISE)."),
        ("카메라워크·촬영",             "촬영 잘 했네요/각도 좋다/조명 좋네 등 카메라·촬영 품질 평가."),
        ("다음 영상 요청·구독·응원",     "다음 영상 기대됩니다/구독 누르고 갑니다/응원합니다 류. 영상 시리즈에 대한 반응."),
        ("영상 길이·구성 코멘트",        "딱 적당한 길이/너무 길어요/챕터 나눠주세요 등 영상 구성에 대한 평가."),
    ],
    "QUESTION": [
        ("성능·스펙 질문",              "이거 게임 잘 돌아가요?/벤치 점수 어느 정도?/발열은 어때요? 등 성능 관련 의문문."),
        ("배터리·충전 질문",            "배터리 몇 시간 가요?/고속충전 되나요?/사용시간 어느 정도? 등."),
        ("가격·구매처 질문",            "어디서 사면 싸요?/지금 사도 되나요?/할인 언제? 등 구매 관련."),
        ("호환·연결 질문",              "이전 모델 케이스 호환되나요?/케이블 USB-C인가요?/페어링 어떻게 하나요? 등."),
        ("기능·옵션 질문",              "방수 되나요?/저장공간 옵션 뭐 있어요?/색상 종류는요? 등."),
        ("비교·대체 질문",              "삼성꺼랑 비교하면?/이거랑 아이폰 중에 뭐가 나아요?/전작 대비 차이는? 등 비교 질문."),
        ("사용·세팅 질문",              "처음 켰을 때 뭐부터 해야 해요?/이 기능 어떻게 끄나요?/추천 설정 있나요? 등."),
        ("문제·고장 질문",              "이런 증상 정상인가요?/AS 어떻게 받나요?/펌업 뒤 느려졌는데? 등 문제 관련."),
    ],
}


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

def generate_synthetic(
    count: int,
    batch_id: str,
    label: str = "NOISE",
    min_length: int = 10,
) -> list[dict]:
    """주어진 label 의 카테고리를 균등 분배해 합성 댓글 생성.

    Args:
        count: 목표 총 건수
        batch_id: 합성 batch 식별자 (video_id prefix 로 사용)
        label: "NOISE" 또는 "VIDEO_REACTION" 등 LABEL_DEFINITIONS 의 키
        min_length: 길이 (글자 수) 하한. NOISE 의 경우 짧은 반응(ㅋㅋ, 와)은
                    이미 학습 데이터에 충분 → dedup 회피 위해 길게 생성/필터.
    """
    if label not in LABEL_DEFINITIONS:
        raise ValueError(
            f"지원하지 않는 라벨: {label}. "
            f"가능: {list(LABEL_DEFINITIONS.keys())}"
        )
    categories = SYNTHETIC_CATEGORIES[label]
    system_prompt = _system_prompt(label, LABEL_DEFINITIONS[label])

    llm = _get_llm(temperature=0.9)  # 다양성 위해 temperature 높임
    per_cat = max(count // len(categories), 5)
    # LLM 이 짧은 거 안 만들도록 보정해서 더 요청 (후처리 필터 손실 보전).
    over_request_factor = 1.8 if min_length >= 8 else 1.2
    per_cat_request = int(per_cat * over_request_factor)
    out: list[dict] = []
    dropped_short = 0
    for cat_name, cat_desc in categories:
        if len(out) >= count:
            break
        length_hint = (
            f" 길이는 최소 {min_length}글자 이상의 자연스러운 문장으로 작성."
            if min_length >= 6 else ""
        )
        user = (
            f"카테고리: {cat_name}\n"
            f"설명: {cat_desc}\n\n"
            f"위 카테고리에 정확히 부합하는 **{label}** 댓글을 정확히 {per_cat_request}개 생성하세요.\n"
            f"다양한 어조·맞춤법 변형을 포함. 실제 유튜브 댓글 같은 자연스러움 유지.{length_hint}\n"
            f"JSON 배열만 출력: [{{\"text\": \"...\"}}, ...]"
        )
        print(f"  [{cat_name}] 요청 중...")
        try:
            arr = _call_llm_json(llm, system_prompt, user)
        except Exception as e:
            print(f"  [{cat_name}] FAIL: {e}")
            continue
        for i, rec in enumerate(arr):
            text = (rec.get("text") or "").strip()
            if not text:
                continue
            if len(text) > C.MAX_TEXT_LEN:
                continue
            if len(text) < max(C.MIN_TEXT_LEN, min_length):
                dropped_short += 1
                continue
            out.append(_make_record(
                text=text,
                comment_id=f"synth-{label[:4]}-{batch_id}-{cat_name[:4]}-{i:03d}",
                video_id=f"synthetic-{label[:4]}-{batch_id}-{cat_name[:4]}",
                confidence=0.95,
                reasoning=f"synthetic {label} ({cat_name})",
                label=label,
            ))
        print(f"  [{cat_name}] 누적 {len(out)}/{count}  (짧아서 drop: {dropped_short})")
        if len(out) >= count:
            out = out[:count]
            break
    if dropped_short:
        print(f"\n총 짧음 drop: {dropped_short}건 (min_length={min_length})")
    return out


# ---------------------------------------------------------------------------
# Label mode — 외부 raw 댓글 라벨링 후 NOISE 만 추출
# ---------------------------------------------------------------------------

def _classify_batch(
    candidates: list[dict],
    target_label: str,
    batch_size: int,
    min_confidence: float,
) -> list[dict]:
    """후보 댓글 배치를 GPT-4.1 로 4-class 분류 후 target_label 만 추출."""
    if not candidates:
        return []
    llm = _get_llm(temperature=0.0)
    out: list[dict] = []
    n_batches = (len(candidates) + batch_size - 1) // batch_size
    for bi, start in enumerate(range(0, len(candidates), batch_size), 1):
        batch = candidates[start:start + batch_size]
        items = "\n".join(f"  {i}. {r['text']}" for i, r in enumerate(batch))
        user = f"분류할 댓글 {len(batch)}개:\n{items}\n\nJSON 배열로 출력."
        try:
            arr = _call_llm_json(llm, CLASSIFY_SYSTEM_PROMPT, user)
        except Exception as e:
            print(f"  batch {bi}/{n_batches} FAIL: {e}")
            continue
        for entry in arr:
            try:
                idx = int(entry["i"])
                label = entry["label"]
                conf = float(entry["confidence"])
            except (KeyError, TypeError, ValueError):
                continue
            if label != target_label:
                continue
            if conf < min_confidence:
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
                label=target_label,
            ))
        print(f"  batch {bi}/{n_batches}  누적 {target_label} 확정: {len(out)}")
    return out


def label_external(
    input_path: Path,
    target_label: str = "NOISE",
    batch_size: int = 25,
    apply_heuristic: bool = True,
    text_field: str = "text",
    video_field: str = "video_id",
) -> list[dict]:
    """raw 댓글 JSONL 을 GPT-4.1 로 분류 → target_label 만 반환."""
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

    if apply_heuristic and target_label == "NOISE":
        before = len(raw)
        raw = [r for r in raw if looks_like_noise(r["text"])]
        print(f"NOISE heuristic pre-filter: {before} -> {len(raw)} candidates")

    return _classify_batch(raw, target_label, batch_size, C.MIN_CONFIDENCE)


# ---------------------------------------------------------------------------
# YouTube fetch mode
# ---------------------------------------------------------------------------

def _video_ids_from_labeled() -> list[str]:
    """기존 labeled_gpt41_azure.jsonl 에서 unique video_id 추출."""
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
            if vid and not str(vid).startswith(("synthetic", "synth-")):
                vids.add(str(vid))
    return sorted(vids)


def _fetch_youtube_comments(video_ids: list[str], per_video: int = 100) -> list[dict]:
    """YouTube Data API v3 — top-level 댓글 fetch. order='time'."""
    import os
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
                status = getattr(e, "resp", None) and e.resp.status
                print(f"  [{vid}] skip (HTTP {status})")
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


def fetch_from_youtube(
    target_label: str,
    video_ids: list[str] | None = None,
    per_video: int = 100,
    max_videos: int = 30,
    target: int = 300,
    batch_size: int = 25,
    apply_heuristic: bool = False,
    seed: int = 42,
    min_length: int = 6,
) -> list[dict]:
    """YouTube 실 댓글 fetch → GPT-4.1 4-class 분류 → target_label 만 추출.

    합성 데이터의 stereotype 문제 우회. 실제 운영 분포에 가까운 댓글만 학습 데이터화.
    min_length: 짧은 NOISE 는 학습 데이터에 이미 충분 → dedup 회피 위해 기본 6+.
    """
    if not video_ids:
        video_ids = _video_ids_from_labeled()
    if not video_ids:
        raise RuntimeError(
            "video_ids 0건. --video-ids 또는 --video-ids-file 로 직접 제공하거나, "
            "comment_labels/labeled_gpt41_azure.jsonl 이 있어야 자동 추출 가능."
        )
    rng = random.Random(seed)
    rng.shuffle(video_ids)
    video_ids = video_ids[:max_videos]
    print(f"video IDs: {len(video_ids)} (max {max_videos})")

    print(f"\n[1] YouTube fetch (~{per_video}/video)")
    raw = _fetch_youtube_comments(video_ids, per_video=per_video)
    print(f"  total fetched: {len(raw)}")

    # 길이 필터 — min_length 적용 (짧은 NOISE 는 dedup 통과 어려움)
    effective_min = max(C.MIN_TEXT_LEN, min_length)
    before = len(raw)
    raw = [r for r in raw if effective_min <= len(r["text"]) <= C.MAX_TEXT_LEN]
    print(f"  after length filter (≥{effective_min}자): {before} → {len(raw)}")
    # 텍스트 중복 제거
    seen: set[str] = set()
    dedup: list[dict] = []
    for r in raw:
        key = r["text"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    print(f"  after length + dedup: {len(dedup)}")
    if not dedup:
        return []

    # 휴리스틱 (NOISE 만 의미 있음, VR 은 적용 안 함)
    candidates = dedup
    if apply_heuristic and target_label == "NOISE":
        before = len(candidates)
        candidates = [r for r in candidates if looks_like_noise(r["text"])]
        print(f"  NOISE heuristic pre-filter: {before} -> {len(candidates)}")

    if not candidates:
        return []

    # GPT-4.1 분류
    print(f"\n[2] GPT-4.1 classification (batch {batch_size})")
    classified = _classify_batch(candidates, target_label, batch_size, C.MIN_CONFIDENCE)
    print(f"  total {target_label} confirmed (conf >= {C.MIN_CONFIDENCE}): {len(classified)}")

    if len(classified) > target:
        # like_count 높은 것 우선 (자연스러운 댓글)
        like_map = {r["comment_id"]: r.get("like_count", 0) for r in dedup}
        classified.sort(key=lambda r: like_map.get(r["comment_id"], 0), reverse=True)
        classified = classified[:target]
        print(f"  trimmed to target {target}")
    return classified


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
    label: str = "NOISE",
) -> dict:
    # final_action / exclusion_reason 은 라벨에 따라 적절히 매핑.
    final_action_map = {
        "NOISE":           ("EXCLUDE",         "NOISE"),
        "VIDEO_REACTION":  ("EXCLUDE",         "VIDEO_REACTION"),
        "PRODUCT_OPINION": ("ANALYZE",         None),
        "QUESTION":        ("AUXILIARY_STORE", None),
    }
    final_action, exclusion_reason = final_action_map.get(label, ("EXCLUDE", label))
    is_product_related = label in {"PRODUCT_OPINION", "QUESTION"}
    return {
        "comment_id": comment_id,
        "video_id": video_id,
        "product_id": None,
        "text": text,
        "label": label,
        "confidence": round(confidence, 4),
        "label_scores": {label: round(confidence, 4)},
        "teacher_model": "openai/gpt-4.1-2025-04-14",
        "final_action": final_action,
        "exclusion_reason": exclusion_reason,
        "is_product_related": is_product_related,
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


def auto_backup_and_apply(records: list[dict], label: str) -> tuple[Path, Path]:
    """마이닝 결과를 자동으로:
       1. comment_labels/backups/<label>_<timestamp>.jsonl 로 timestamped 백업
       2. comment_labels/labeled_gpt41_azure.jsonl 에 직접 append
    Returns: (backup_path, labeled_path)
    """
    if not records:
        raise ValueError("records 0건 — apply 대상 없음")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backups_dir = REPO_ROOT / "comment_labels" / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backups_dir / f"{label.lower()}_{ts}_n{len(records)}.jsonl"
    write_jsonl(backup_path, records, append=False)

    labeled_path = REPO_ROOT / "comment_labels" / "labeled_gpt41_azure.jsonl"
    # 안전: labeled.jsonl 이 이미 있을 때만 append (없으면 사용자 의도 확인 필요)
    if not labeled_path.exists():
        raise FileNotFoundError(
            f"labeled.jsonl 없음 — apply 불가: {labeled_path}\n"
            f"백업만 완료: {backup_path}"
        )
    write_jsonl(labeled_path, records, append=True)
    return backup_path, labeled_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["synthetic", "label", "youtube"], required=True,
                    help="synthetic=GPT 합성, label=JSONL 분류, youtube=YouTube fetch+분류")
    ap.add_argument("--label", default="NOISE",
                    help="추출할 라벨. NOISE / VIDEO_REACTION / PRODUCT_OPINION / QUESTION")
    ap.add_argument("--count", type=int, default=500,
                    help="(synthetic) 생성 댓글 수")
    ap.add_argument("--input", type=str, default=None,
                    help="(label) raw 댓글 JSONL 경로 (text 필드 필수)")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--video-field", default="video_id")
    ap.add_argument("--no-heuristic", action="store_true",
                    help="(label/youtube) NOISE 휴리스틱 pre-filter 건너뛰기")
    ap.add_argument("--batch-size", type=int, default=25)
    # youtube 모드 옵션
    ap.add_argument("--video-ids", type=str, default=None,
                    help="(youtube) 콤마로 구분된 video ID 리스트")
    ap.add_argument("--video-ids-file", type=str, default=None,
                    help="(youtube) 한 줄에 하나씩 video ID 가 있는 파일")
    ap.add_argument("--per-video", type=int, default=100,
                    help="(youtube) 영상 당 fetch 댓글 수")
    ap.add_argument("--max-videos", type=int, default=30,
                    help="(youtube) 처리할 영상 최대 개수")
    ap.add_argument("--target", type=int, default=300,
                    help="(youtube) 목표 라벨 추출 건수")
    ap.add_argument("--min-length", type=int, default=None,
                    help="텍스트 최소 길이 (글자 수). 기본: synthetic=10, youtube=6. "
                         "짧은 NOISE 는 학습 데이터에 이미 충분해 dedup 에 잘림.")
    # 출력
    ap.add_argument("--output", type=str, default=None,
                    help="기본: comment_labels/labeled_gpt41_azure_<label>_extra.jsonl")
    ap.add_argument("--append", action="store_true",
                    help="기존 출력 파일에 append (기본은 덮어쓰기)")
    ap.add_argument("--apply", action="store_true",
                    help="★ 마이닝 결과를 즉시 labeled_gpt41_azure.jsonl 에 append + "
                         "comment_labels/backups/<label>_<ts>.jsonl 로 자동 timestamped 백업. "
                         "다음 단계로 바로 prepare_dataset → train 가능.")
    args = ap.parse_args()

    # synthetic 모드는 LABEL_DEFINITIONS 에 있는 라벨만 가능
    if args.mode == "synthetic" and args.label not in LABEL_DEFINITIONS:
        ap.error(f"synthetic 모드는 {list(LABEL_DEFINITIONS.keys())} 만 지원. "
                 f"VIDEO_REACTION 합성은 권장 안 함 (stereotype 위험). "
                 f"VR 마이닝은 --mode youtube --label VIDEO_REACTION 권장.")

    # output path: label 별 기본 파일명
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = args.label.lower()
        output_path = REPO_ROOT / "comment_labels" / f"labeled_gpt41_azure_{suffix}_extra.jsonl"

    if args.mode == "synthetic":
        batch_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        min_len = args.min_length if args.min_length is not None else 10
        print(f"[mode=synthetic] label={args.label} count={args.count} "
              f"min_length={min_len} batch_id={batch_id}")
        recs = generate_synthetic(count=args.count, batch_id=batch_id,
                                   label=args.label, min_length=min_len)
    elif args.mode == "label":
        if not args.input:
            ap.error("--mode label 사용 시 --input 필수")
        print(f"[mode=label] input={args.input} target_label={args.label} "
              f"heuristic={not args.no_heuristic}")
        recs = label_external(
            input_path=Path(args.input),
            target_label=args.label,
            batch_size=args.batch_size,
            apply_heuristic=not args.no_heuristic,
            text_field=args.text_field,
            video_field=args.video_field,
        )
    else:  # youtube
        # video_ids 소스 결정
        if args.video_ids:
            vids = [v.strip() for v in args.video_ids.split(",") if v.strip()]
            src_desc = "CLI --video-ids"
        elif args.video_ids_file:
            with open(args.video_ids_file, encoding="utf-8") as f:
                vids = [line.strip() for line in f if line.strip()]
            src_desc = f"file {args.video_ids_file}"
        else:
            vids = None  # fetch_from_youtube 에서 labeled.jsonl 에서 자동
            src_desc = "labeled_gpt41_azure.jsonl (auto)"

        min_len = args.min_length if args.min_length is not None else 6
        print(f"[mode=youtube] target_label={args.label} target={args.target} "
              f"min_length={min_len}")
        print(f"  source: {src_desc}")
        print(f"  max_videos={args.max_videos}, per_video={args.per_video}")
        recs = fetch_from_youtube(
            target_label=args.label,
            video_ids=vids,
            per_video=args.per_video,
            max_videos=args.max_videos,
            target=args.target,
            batch_size=args.batch_size,
            apply_heuristic=not args.no_heuristic,
            min_length=min_len,
        )

    if not recs:
        print("생성된 레코드 0건. 출력 파일 갱신 안 함.")
        return

    write_jsonl(output_path, recs, append=args.append)
    print(f"\n총 {len(recs)}건 {args.label} → {output_path} ({'append' if args.append else 'overwrite'})")

    if args.apply:
        try:
            backup_path, labeled_path = auto_backup_and_apply(recs, args.label)
            print()
            print(f"✓ 자동 백업      : {backup_path}")
            print(f"✓ 자동 합치기     : {labeled_path} (append {len(recs)} 건)")
            print()
            print("다음 단계:")
            print("  python -m local_classifier.prepare_dataset")
            print("  python -m local_classifier.train")
            print("  python -m local_classifier.evaluate")
        except Exception as e:
            print(f"\n[apply 실패] {e}")
    else:
        print()
        print("학습 데이터에 합치려면 (수동) — 또는 --apply 플래그 사용:")
        print(f"  cat {output_path} >> {REPO_ROOT}/comment_labels/labeled_gpt41_azure.jsonl")
        print(f"  python -m local_classifier.prepare_dataset")
        print(f"  python -m local_classifier.train")


if __name__ == "__main__":
    main()
