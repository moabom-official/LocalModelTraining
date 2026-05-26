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

# ---- Labels (4-class direct training) --------------------------------------
# 단일 4-class 학습 — NOISE 를 양성 클래스로 직접 학습.
# 진화 히스토리:
#   - v1 5-class : PO / VR / CHATTER / Q / OFF_TOPIC
#   - v2 4-class : PO / VR / Q / NOISE  (CHATTER + OFF_TOPIC 통합)  ← 현재
#   - v3 3-class : NOISE 를 학습에서 제외 + softmax rejection      → NOISE F1=0.034 실패
#   - v4 3-class : 동일 구조 + sigmoid BCE                          → NOISE F1=0.0 더 실패
# 결론: NOISE 도 양성 클래스로 직접 학습이 가장 안정적. NOISE F1=0.60 정체는
# 데이터 보강(mine_noise.py / fetch_noise_groq.py)으로 해결.

LABEL_NAMES = [
    "PRODUCT_OPINION",
    "VIDEO_REACTION",
    "QUESTION",
    "NOISE",
]
LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}
ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}
NUM_LABELS = len(LABEL_NAMES)
NOISE_LABEL = "NOISE"

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
