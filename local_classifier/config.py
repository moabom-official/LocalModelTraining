"""local_classifier config — paths, label map, hyperparameters.

Defaults tuned for **NVIDIA A40 (48GB, Ampere sm_86)** server training.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = REPO_ROOT / "comment_labels"
INPUT_JSONL = LABELS_DIR / "labeled_gpt41_azure.jsonl"

OUTPUT_DIR = Path(
    os.environ.get(
        "LOCAL_CLASSIFIER_OUTPUT",
        str(REPO_ROOT / "local_classifier" / "artifacts"),
    )
)
DATA_DIR = OUTPUT_DIR / "data"
MODEL_DIR = OUTPUT_DIR / "model"
LOG_DIR = OUTPUT_DIR / "logs"

# ---- Labels (dual scheme: 3-class multi-label training, 4-class output) ----
# Strategy: NOISE 는 양성 클래스로 학습하지 않고, 추론 시 **per-class sigmoid**
# 최댓값이 REJECTION_THRESHOLD 미만이면 NOISE 로 분류. 2026-05-26 변경 —
# softmax + threshold 접근은 OOD overconfidence 로 NOISE F1=0.034 까지 떨어짐.
# softmax 는 확률 합=1 강제 → 모델에 "셋 다 아님" 출구가 없음.
# 해결: per-class **sigmoid** head + BCE loss → 각 클래스 독립 확률 →
# 셋 다 < 임계값이면 NOISE.

# 모델이 학습하는 양성 클래스 (3개)
TRAINING_LABEL_NAMES = [
    "PRODUCT_OPINION",
    "VIDEO_REACTION",
    "QUESTION",
]
TRAINING_LABEL2ID = {name: i for i, name in enumerate(TRAINING_LABEL_NAMES)}
TRAINING_ID2LABEL = {i: name for i, name in enumerate(TRAINING_LABEL_NAMES)}
NUM_LABELS = len(TRAINING_LABEL_NAMES)  # 모델 head 크기 (3)

# 다운스트림(comment_filtering_agent / DB enum) 에 노출되는 4-class 라벨.
# evaluate / 운영 추론 시 NOISE 가 합쳐진 형태로 보고됨.
NOISE_LABEL = "NOISE"
OUTPUT_LABEL_NAMES = TRAINING_LABEL_NAMES + [NOISE_LABEL]
OUTPUT_LABEL2ID = {name: i for i, name in enumerate(OUTPUT_LABEL_NAMES)}
OUTPUT_ID2LABEL = {i: name for i, name in enumerate(OUTPUT_LABEL_NAMES)}

# 하위 호환 alias — 기존 코드/문서가 LABEL_NAMES / LABEL2ID / ID2LABEL
# 를 참조하는 경우 4-class 출력 라벨을 가리킴.
LABEL_NAMES = OUTPUT_LABEL_NAMES
LABEL2ID = OUTPUT_LABEL2ID
ID2LABEL = OUTPUT_ID2LABEL

# 입력 라벨이 구 5-class 체계(CHATTER / OFF_TOPIC)로 들어오면 자동으로 NOISE 로 통합.
LEGACY_LABEL_REMAP = {
    "CHATTER": "NOISE",
    "OFF_TOPIC": "NOISE",
}


def remap_legacy_label(label: str | None) -> str | None:
    """구 5-class 라벨(CHATTER / OFF_TOPIC)을 NOISE 로 자동 통합.

    None 입력은 None 반환. 알 수 없는 라벨은 그대로 통과 → 후속 membership
    체크 단계에서 제외됨.
    """
    if label is None:
        return None
    return LEGACY_LABEL_REMAP.get(label, label)


# ---- Rejection (open-set classification, sigmoid head) ---------------------
# 추론 시 per-class **sigmoid** 최댓값이 이 값 미만이면 NOISE 로 분류.
# softmax 와 의미가 다름 — sigmoid 0.5 = "이 클래스일 확률 50%". 셋 다 <τ면
# 모델이 "어느 것도 충분히 확신 못 함" 을 표현 가능 (softmax 는 합=1 제약).
# 운영 데이터 calibration 후 tune. baseline 0.5.
REJECTION_THRESHOLD = 0.5

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
BASE_MODEL = "klue/roberta-large"       # 340M; swap back to roberta-base to compare
MAX_SEQ_LEN = 128

# ---- Training (A40 48GB) ---------------------------------------------------
TRAIN_BATCH_SIZE = 32                   # large@128 seq — safe on A40, no OOM
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 1e-5                    # RoBERTa-large standard; 2e-5 often unstable
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
NUM_EPOCHS = 20
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
