# LocalModelTraining

모아봄 댓글 필터링 Agent 의 **로컬 KLUE-RoBERTa 분류기 학습 코드**.

GPT-4.1 teacher 라벨로 `klue/roberta-large` 를 distill 하고, 운영에서는 cascade router (`local → 저신뢰만 GPT-4.1 fallback`) 로 API 호출비를 줄이기 위한 분리 리포지토리.

## 라벨 체계 (3-class training + rejection → 4-class output, 2026-05-25)

**모델은 3개 양성 클래스만 학습**하고, 추론 시 `max softmax < REJECTION_THRESHOLD` 면 NOISE 로 분류 (classification with rejection / open-set 접근).

| 학습 head id | 학습 라벨 | 설명 |
|---:|-------|------|
| 0 | PRODUCT_OPINION | 제품 평가 |
| 1 | VIDEO_REACTION | 영상·리뷰어 반응 |
| 2 | QUESTION | 제품 관련 질문 |

| 출력 라벨 (4-class, downstream) | 산출 방식 |
|---|---|
| PRODUCT_OPINION / VIDEO_REACTION / QUESTION | argmax (단, max softmax ≥ τ) |
| **NOISE** | max softmax < τ → reject |

**왜 이 구조?** 첫 시도는 4-class 직접 학습이었으나 NOISE F1=0.60 에 막힘 (test macro F1 0.775). 원인: CHATTER(짧고 의미없음) + OFF_TOPIC(다른 주제) 가 의미적으로 너무 이질적이라 단일 클래스로 학습하기 어려움. 3-class 학습은 깨끗한 양성 클래스만 모델링하고 NOISE 는 "정답 3개 중 어느 것도 아닌 것" 으로 정의.

운영 export 가 구 5-class (CHATTER / OFF_TOPIC) 로 들어오면 `local_classifier/config.py:remap_legacy_label()` 가 자동으로 NOISE 로 통합한다. NOISE 라벨은 train/val 에서 제거되고 test 에만 유지되어 rejection 성능 평가에 사용됨.

## 리포 구조

```
LocalModelTraining/
├── local_classifier/      # 학습 파이프라인 (config / preprocess / dataset / train / evaluate)
│   ├── config.py
│   ├── preprocess.py
│   ├── prepare_dataset.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── classifier.py      # 추론 wrapper (BaseCommentClassifier 호환)
│   ├── router.py          # CascadeRouter (local → API fallback)
│   ├── shadow.py          # ShadowLogger
│   ├── README.md          # 상세 사용 가이드
│   └── requirements.txt
└── comment_labels/        # GPT-4.1 teacher 라벨 (6,375건)
    ├── README.md
    ├── labeled_gpt41_azure.jsonl
    └── labeled_gpt41_azure.csv
```

## 빠른 시작

```bash
# 1) 의존성 (CUDA 12.x A40 기준)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r local_classifier/requirements.txt

# 2) 데이터셋 준비 (구 5-class 라벨은 자동 매핑됨)
python -m local_classifier.prepare_dataset

# 3) 학습 (A40 권장, bf16 autocast)
python -m local_classifier.train

# 4) 평가
python -m local_classifier.evaluate
```

산출물은 `local_classifier/artifacts/` 에 떨어지며 `.gitignore` 처리됨. 자세한 학습 옵션·임계값 튜닝 가이드는 [`local_classifier/README.md`](local_classifier/README.md) 참고.

## 본 리포 vs Moabom_Prototype

본 리포는 학습 전용으로 분리되었지만, 코드/라벨 매핑은 `Moabom_Prototype/comment_filtering_agent/classifiers/` 와 1:1 호환된다 (NR-007/012: 모델 교체 시 기존 시스템 수정 최소화).
