"""local_classifier config — paths, label map, hyperparameters.

Defaults tuned for **NVIDIA A40 (48GB, Ampere sm_86)** server training.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = REPO_ROOT / "comment_labels"
INPUT_JSONL = LABELS_DIR / "labeled_gpt41_azure.jsonl"

_ARTIFACTS_ROOT = Path(
    os.environ.get(
        "LOCAL_CLASSIFIER_OUTPUT",
        str(REPO_ROOT / "local_classifier" / "artifacts"),
    )
)
# 라벨 체계별로 산출물 분리 (4_labels / 3_labels 비교 시 덮어쓰기 방지).
# LABEL_SCHEME 환경변수로 override 가능.
LABEL_SCHEME = os.environ.get("LABEL_SCHEME", "3_labels")
OUTPUT_DIR = _ARTIFACTS_ROOT / LABEL_SCHEME
DATA_DIR = OUTPUT_DIR / "data"      # 모델 공유 — 전처리 결과는 모델 무관 (라벨 체계 종속)

# ---- Labels (3-class direct training, NOISE → VIDEO_REACTION 통합) ---------
# 단일 3-class 학습. NOISE / CHATTER / OFF_TOPIC 모두 VIDEO_REACTION 으로 흡수.
# 근거: NOISE 와 VR 모두 agent decision 에서 EXCLUDE 액션. 운영 구분 가치 낮음.
# 4-class 학습 시 NOISE F1=0.667 가 macro 끌어내림 → 통합으로 metric 개선.
# 진화 히스토리:
#   - v1 5-class : PO / VR / CHATTER / Q / OFF_TOPIC               → baseline
#   - v2 4-class : PO / VR / Q / NOISE (CHATTER+OFF_TOPIC 통합)    → macro 0.79
#   - v3 3-class rejection 실험들 (softmax / sigmoid BCE)          → NOISE F1=0.0 실패
#   - v4 4-class + 데이터 보강 (NOISE 484→541)                     → macro 0.795 (best)
#   - v5 3-class : NOISE → VIDEO_REACTION 통합                      ← 현재
# 운영 (comment_filtering_agent / DB enum) 은 4-class 그대로 유지. 분류기 출력만 3-class.

LABEL_NAMES = [
    "PRODUCT_OPINION",
    "VIDEO_REACTION",   # 구 NOISE / CHATTER / OFF_TOPIC 통합 흡수
    "QUESTION",
]
LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}
ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}
NUM_LABELS = len(LABEL_NAMES)
NOISE_LABEL = "NOISE"  # mine_noise.py 호환용 상수 (학습엔 안 쓰이고 remap 됨)

# 구 NOISE / CHATTER / OFF_TOPIC 라벨이 입력으로 들어오면 자동으로 VR 로 흡수.
# 4-class 시절 마이닝한 jsonl 그대로 사용 가능.
LEGACY_LABEL_REMAP = {
    "CHATTER":   "VIDEO_REACTION",
    "OFF_TOPIC": "VIDEO_REACTION",
    "NOISE":     "VIDEO_REACTION",
}


def remap_legacy_label(label: str | None) -> str | None:
    """구 NOISE / CHATTER / OFF_TOPIC 을 VIDEO_REACTION 으로 자동 흡수.

    None 입력은 None 반환. 알 수 없는 라벨은 그대로 통과 → 후속 membership
    체크 단계에서 제외됨.
    """
    if label is None:
        return None
    return LEGACY_LABEL_REMAP.get(label, label)

# ---- Preprocess filters ----------------------------------------------------
MIN_CONFIDENCE = 0.85
DROP_TEACHERS = {"gpt-4.1-mini"}      # mini export held out as OOD-eval source
ALLOWED_LANGS = {"ko", "en"}           # others are kept but flagged in stats
MIN_TEXT_LEN = 2
MAX_TEXT_LEN = 1000

# ---- Split (video_id grouped) ----------------------------------------------
VAL_RATIO = 0.10
TEST_RATIO = 0.10
SEED = 42

# ---- Model -----------------------------------------------------------------
# BASE_MODEL 환경변수로 override 가능 → 같은 데이터로 여러 모델 비교 학습.
# 한국어 옵션:
#   klue/roberta-large                   (340M, KLUE팀, 기본)
#   klue/roberta-base                    (110M)
#   team-lucid/deberta-v3-base-korean    (180M, Korean DeBERTa-v3)
#   microsoft/mdeberta-v3-base           (280M, multilingual DeBERTa)
BASE_MODEL = os.environ.get("BASE_MODEL", "klue/roberta-large")
MAX_SEQ_LEN = 128

# 모델별 산출물 디렉토리 분리 (artifacts/<slug>/model|logs/).
# DATA_DIR 은 공유 — 같은 split 으로 fair comparison.
MODEL_SLUG = BASE_MODEL.replace("/", "__").replace(":", "_")
MODEL_DIR = OUTPUT_DIR / MODEL_SLUG / "model"
LOG_DIR = OUTPUT_DIR / MODEL_SLUG / "logs"

# ---- Training (A40 48GB) ---------------------------------------------------
TRAIN_BATCH_SIZE = 32                   # large@128 seq — safe on A40, no OOM
EVAL_BATCH_SIZE = 64
# 모델별 권장: RoBERTa-large 1e-5, RoBERTa-base 2e-5, DeBERTa-v3 2~3e-5.
# LEARNING_RATE 환경변수로 override 가능 → 모델별 fine-tuning.
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-5"))
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
NUM_EPOCHS = 4                          # best val_F1 가 epoch 4 에서 plateau (관측 기반)
LABEL_SMOOTHING = 0.05
USE_BF16 = True                         # A40 native bf16 — prefer over fp16
USE_CLASS_WEIGHTS = True
CONFIDENCE_AS_WEIGHT = True
GRADIENT_CLIP = 1.0
DATALOADER_NUM_WORKERS = 4              # server CPU usually has cores to spare
PIN_MEMORY = True
LOG_EVERY_N_STEPS = 50

# ---- Router (cascade) ------------------------------------------------------
ROUTER_TAU_HIGH = 0.85                  # local accept ≥
ROUTER_TAU_LOW = 0.55                   # below this also flagged as disagreement
