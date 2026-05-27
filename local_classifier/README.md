# KLUE-RoBERTa Distill — 3-class 댓글 분류기

GPT-4.1 teacher 라벨(`comment_labels/labeled_gpt41_azure.jsonl`, 6,375건)로 `klue/roberta-large`를 distill하고, 운영에서는 confidence-gated cascade(`local 1차 → 저신뢰만 GPT-4.1 fallback`)로 호출비를 줄이는 모듈.

라벨: **PRODUCT_OPINION / VIDEO_REACTION / QUESTION** (3-class). softmax + argmax 직접 학습.

> **NOISE 통합 (2026-05-27)**: 구 NOISE / CHATTER / OFF_TOPIC 라벨은 모두 **VIDEO_REACTION** 으로 흡수. 근거:
> - agent decision engine 에서 NOISE 와 VR 모두 `EXCLUDE` 액션 — 운영 구분 가치 낮음
> - 4-class 학습 시 NOISE F1=0.667 가 macro 끌어내림 (병목)
> - 통합 후 학습 데이터 VR 835 → 1319 (+58%) → 학습 신호 강화
>
> 운영 (`comment_filtering_agent` / DB enum) 은 4-class 그대로 유지. 분류기 출력만 3-class.

> **설계 진화 히스토리 (참고)**
> 1. 5-class 직접 학습 (CHATTER / OFF_TOPIC 분리) → baseline
> 2. **4-class 통합 학습** (NOISE = CHATTER + OFF_TOPIC) → test macro F1 **0.775** ← **현재 채택**
> 3. 3-class softmax + threshold rejection → val F1 0.90 이지만 test NOISE F1 0.034 (OOD overconfidence) → 폐기
> 4. 3-class multi-label sigmoid + BCE → val F1 0.89 / test NOISE F1 0.0 더 악화 → 폐기
>
> **결론**: 후처리 rejection 으로는 OOD 검출 한계. NOISE 도 양성 클래스로 직접 학습이 가장 안정적이며, 약한 클래스(NOISE / VR) F1 정체는 데이터 보강 (`mine_noise.py` — GPT-4.1 합성, label-aware) 으로 해결.
>
> 구 CHATTER / OFF_TOPIC 라벨 입력은 `config.remap_legacy_label()` 가 NOISE 로 자동 매핑.

## 폴더 구조

```
local_classifier/
├── config.py              # 경로·라벨·하이퍼파라미터 (A40 기준 batch/bf16)
├── preprocess.py          # PII scrub · NFKC · 반복문자 · lang detect · dedup key
├── prepare_dataset.py     # JSONL → 정제 → video_id 그룹 split → JSONL + class_weights
├── dataset.py             # torch Dataset
├── train.py               # KLUE-RoBERTa-base fine-tune (bf16, class+conf weighted CE)
├── evaluate.py            # test acc · per-class P/R/F1 · macro-F1 · confusion matrix
├── classifier.py          # LocalRobertaClassifier (BaseCommentClassifier 호환)
├── router.py              # CascadeRouter (local → API fallback)
├── shadow.py              # ShadowLogger (운영 영향 없이 disagreement 수집)
├── requirements.txt
└── artifacts/             # gitignore — data/, model/best/, logs/
```

## 라벨 매핑 (고정 — 변경 시 재학습 필수)

`comment_filtering_agent/classifiers/classifier_interface.py:FineTunedClassifier` 와 동일 순서. **4-class** (2026-05-25 통합).

| id | label | 설명 |
|----|-------|------|
| 0 | PRODUCT_OPINION | 제품 평가 |
| 1 | VIDEO_REACTION | 영상·리뷰어 반응 |
| 2 | QUESTION | 제품 관련 질문 |
| 3 | NOISE | 잡담·밈·제품 무관 (구 CHATTER + OFF_TOPIC 통합) |

### Legacy 라벨 자동 매핑

데이터 파일이 구 5-class (CHATTER / OFF_TOPIC) 로 export 되어 있어도 학습 데이터 전처리 단계에서 자동으로 NOISE 로 변환된다.

```python
from local_classifier.config import remap_legacy_label
remap_legacy_label("CHATTER")    # → "NOISE"
remap_legacy_label("OFF_TOPIC")  # → "NOISE"
remap_legacy_label("QUESTION")   # → "QUESTION" (unchanged)
```

`prepare_dataset.py` 실행 시 변환된 record 수가 콘솔에 출력된다.

운영 라벨 분포 (운영 export 기준): CHATTER 405 + OFF_TOPIC 245 = **NOISE 650 (10%)**.

## 모델 비교 (BASE_MODEL 환경변수)

`BASE_MODEL` 환경변수로 backbone 갈아끼우면 같은 데이터 split 으로 여러 모델 비교 학습 가능. 출력은 `artifacts/<slug>/{model,logs}/` 로 자동 분리됨.

```bash
# 1) 데이터 준비 (한 번만, 모델 무관)
python -m local_classifier.prepare_dataset

# 2-A) KLUE-RoBERTa-large (기본)
python -m local_classifier.train
python -m local_classifier.evaluate

# 2-B) Korean DeBERTa-v3-base — DeBERTa 는 lr 2~3e-5 필요 (RoBERTa 1e-5 와 다름)
LEARNING_RATE=2e-5 BASE_MODEL=team-lucid/deberta-v3-base-korean python -m local_classifier.train
BASE_MODEL=team-lucid/deberta-v3-base-korean python -m local_classifier.evaluate

# 2-C) (선택) Multilingual DeBERTa-v3
LEARNING_RATE=2e-5 BASE_MODEL=microsoft/mdeberta-v3-base python -m local_classifier.train
BASE_MODEL=microsoft/mdeberta-v3-base python -m local_classifier.evaluate

# 3) 비교 표 출력
python -m local_classifier.compare_models
```

결과는 `artifacts/` 에 모델별로 떨어짐:
```
artifacts/
├── data/                                       # 공유 (전처리 결과)
├── klue__roberta-large/
│   ├── model/best/                            # 학습된 가중치
│   └── logs/{train_metrics,test_summary}.json
├── team-lucid__deberta-v3-base-korean/
│   └── ...
└── microsoft__mdeberta-v3-base/
    └── ...
```

권장 비교 후보 (한국어):

| Model | Params | 특징 |
|---|---:|---|
| `klue/roberta-large` | 340M | KLUE팀, 현재 기본 |
| `klue/roberta-base` | 110M | 3× 빠름, -1~2%p |
| `team-lucid/deberta-v3-base-korean` | 180M | 한국어 DeBERTa-v3, disentangled attention |
| `microsoft/mdeberta-v3-base` | 280M | 다국어 DeBERTa, 한국어 단일보다 약함 |

학습 hyperparameter 는 공통 — fair comparison. DeBERTa-v3 가 RoBERTa 대비 lr 살짝 더 높이면 (2e-5) 좋을 수 있으나 1차 비교에서는 동일 1e-5.

## 환경

NVIDIA **A40 (48GB, Ampere sm_86)** 서버 기준.

```bash
# CUDA 12.x 환경
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r local_classifier/requirements.txt
```

A40에서는 bf16 autocast + TF32 matmul이 기본값 — `config.USE_BF16 = True`.

## 사용 흐름

### 1) 데이터셋 생성

```bash
python -m local_classifier.prepare_dataset
```

처리:
- `teacher_model == "gpt-4.1-mini"` 제외 (119건 → OOD 평가 보관용)
- `confidence < 0.85` 제외
- PII scrub: 이메일·URL·멘션·전화 → 마스크 토큰
- NFKC 정규화 + `ㅋㅋㅋㅋㅋ` 3자 압축 + 공백 정리
- 텍스트 alnum 정규화 후 md5 near-dup dedup
- **video_id 단위** train 80% / val 10% / test 10% 그룹 split (댓글 단위 누설 차단)
- `class_weights.json` (train 빈도 역수)

산출: `artifacts/data/{train,val,test}.jsonl` + `class_weights.json`

### 2) 학습

```bash
python -m local_classifier.train
```

- 모델: `klue/roberta-base` (A40에서 `klue/roberta-large` 도 batch 32~64로 가능)
- 손실: `cross_entropy(logits, labels, weight=class_w, label_smoothing=0.05)` × `per_example_confidence`
- AdamW, lr 2e-5, weight decay 0.01, warmup 10%, 5 epoch, grad clip 1.0
- bf16 autocast + TF32 matmul + cudnn.benchmark
- best val acc 갱신 시 `artifacts/model/best/`에 자동 저장

### 3) 평가

```bash
python -m local_classifier.evaluate
```

`artifacts/logs/test_summary.json` · `test_predictions.jsonl` 산출.
KPI 목표: **macro F1 ≥ 0.85** (GPT-4.1 대비 -3%p 이내).

### 4) 인퍼런스 / 캐스케이드 라우터

```python
from local_classifier.classifier import LocalRobertaClassifier
from local_classifier.router import CascadeRouter
from comment_filtering_agent.classifiers.optimized_batch_classifier import (
    OptimizedBatchClassifier,
)

local = LocalRobertaClassifier(use_gpu=True)
api = OptimizedBatchClassifier(...)        # 기존 GPT-4.1 분류기 그대로
router = CascadeRouter(local=local, api=api, tau_high=0.85, tau_low=0.55)

results = router.classify_batch(comments)  # results[i].classifier_used 로 라우팅 확인
print(router.get_stats())                  # local_rate, api_rate, low_conf_rate
```

### 5) Shadow 모드 (P2 — τ 튜닝용)

운영 출력은 API 그대로, 로컬은 비교 로깅만.

```python
from local_classifier.shadow import ShadowLogger

shadow = ShadowLogger(api_classifier=api, local_classifier=local)
results = shadow.classify_batch(comments)  # API 결과 반환
print(shadow.get_stats())                   # agreement_rate
# artifacts/logs/shadow.jsonl 에 라인별 비교
```

## 임계값 튜닝 가이드

1. shadow 모드로 5~10k건 로그 수집.
2. 로컬 conf 분포 vs (api == local) 일치율 곡선 그리기.
3. `tau_high` = **api 호출 ≤ 20%** 가 되는 최소값.
4. `tau_low` = local-api disagreement 비율이 5%를 넘는 conf 지점.
5. canary 10% 적용 → KPI 미달 시 즉시 롤백.

## 비기능 요구사항

- 라벨 매핑·인터페이스(`ClassificationResult`)는 `comment_filtering_agent` 와 1:1 호환 — 기존 파이프라인 수정 최소화 (NR-007/012).
- 학습·인퍼런스가 분리되어 모델 교체 시 `MODEL_DIR/best/` 만 갈아끼우면 됨.
- shadow 실패는 prod에 영향 없음 (예외 잡고 API 결과만 반환).
