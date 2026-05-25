# 운영(Azure PG) 라벨 댓글 export 요약 (2026-05-20)

- **출처**: Azure PostgreSQL Flexible Server `pg-moabom.postgres.database.azure.com` / db `techdb`
- **Teacher 모델**: 98% `openai/gpt-4.1-2025-04-14` (full), 2% legacy `gpt-4.1-mini`
- **라벨된 댓글**: **6,375** / 운영 raw 60,330 / rule_filter PASS 57,292
  - → PASS 중 11%만 LLM 분류 (Multi-Criteria 컷)
- **영상 분포**: 355개 영상, 153 제품 (라벨된 부분 집합)

## 클래스 분포 (운영)
| label | n | % | avg_conf |
|---|---:|---:|---:|
| PRODUCT_OPINION | 3,824 | 60% | 0.944 |
| VIDEO_REACTION  | 1,087 | 17% | 0.943 |
| QUESTION        |   814 | 13% | 0.962 |
| CHATTER         |   405 |  6% | 0.933 |
| OFF_TOPIC       |   245 |  4% | 0.947 |

## 파일
- `labeled_gpt41_azure.jsonl` (6,375 줄) — full 한 줄 JSON
- `labeled_gpt41_azure.csv`   (6,375 줄) — UTF-8 BOM, QUOTE_ALL, label_scores 5컬럼 flatten
- (이전 `labeled_gpt41mini.*` / `unlabeled_pool.*` 로컬 85건은 폐기됨 — 운영과 격차 75배)

## JSONL/CSV 한 줄 스키마
comment_id, video_id, product_id, text, label, confidence, label_scores(jsonl만)/score_<class>(csv),
teacher_model, final_action(ANALYZE/EXCLUDE/AUXILIARY_STORE), exclusion_reason,
is_product_related, like_count, reply_count, classified_at, reasoning

## 학습 전 권장 처리
1. `teacher_model` 로 mini(119) 분리 — full GPT-4.1만으로 1차 학습
2. **video_id 단위** train/val/test split (355개 영상 → 데이터 누수 차단)
3. OFF_TOPIC 245 최소 클래스 → stratified, class-weight 또는 focal
4. 중복/근접중복 dedup (동일 author 연속 댓글 등 점검)
