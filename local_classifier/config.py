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

# ---- Labels (dual scheme: 3-class training, 4-class output) ----------------
# Strategy: NOISE 는 양성 클래스로 학습하지 않고, 추론 시 max softmax 가
# REJECTION_THRESHOLD 미만이면 NOISE 로 분류 (classification with rejection /
# open-set 접근). 2026-05-25 변경 — 4-class 학습이 NOISE F1=0.60 에 막혀서
# 이질적 데이터 직접 학습을 피하기 위함.

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


# ---- Rejection (open-set classification) -----------------------------------
# 모델 추론 시 max softmax 가 이 값 미만이면 NOISE 로 분류.
# 운영 데이터로 calibration 후 tune. baseline 0.55 (val NOISE F1 기준).
REJECTION_THRESHOLD = 0.55

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
