# 학습 데이터 샘플링 계획

## 1) 목표

- AI Hub 기반 데이터에서 학습/검증/테스트 세트를 안정적으로 구성한다.
- `span`, `complexity` 분포를 반영해 학습 성능 편향을 줄인다.
- 반복 샘플링 시 중복 선택을 방지하고, 유사 샘플 과다 포함을 방지한다.

## 2) 데이터 구성

| Split | 샘플 수 |
|---|---:|
| train | 10,000 |
| validation | 1,000 |
| test | 500 |
| total | 11,500 |

### 초기 학습 테스트용 샘플(별도)

- 목적: 초기 학습 파이프라인이 정상 동작하는지 빠르게 검증
- 별도 저장 수량: `2,000`개
- 운영 원칙:
  - 본 학습/검증/테스트 split과 분리 관리
  - 샘플링 이력(`run_id`, `sample_id`) 기록으로 중복 재사용 추적

## 3) 샘플링 대상/분석 기준

- 데이터 소스: `ai_hub` 원천 데이터
- 분석 단위: 데이터 전체(전수) 대상
- 핵심 분석 축:
  - `span` 관련 통계
  - `complexity`(복잡도) 점수/등급

## 4) 샘플링 원칙

1. 복잡도 균형 샘플링
- 매우 높은 복잡도부터 낮은 복잡도까지 사전에 정의한 비율로 샘플링한다.
- 기본 운영 비율(복잡도 강화): `high 70% / mid 20% / low 10%`
- 동일 복잡도 구간 내부에서는 `complexity_score` 내림차순 우선 선택한다.
- 실제 분포와 모델 목표에 따라 비율은 고정 값으로 확정 후 운영한다.

2. 전수 인덱스 기반 추출
- 전체 데이터를 고유 ID로 인덱싱하고 `sampled_history`를 유지한다.
- 한 번 선택된 샘플은 동일 실험군에서 재선택되지 않도록 상태를 기록한다.
- 권장 메타 필드: `sample_id`, `source`, `complexity_bin`, `span_stats`, `split`, `selected_at`, `run_id`

3. 유사 샘플 중복 필터링
- 구조/레이아웃/텍스트 패턴이 유사한 샘플은 중복으로 간주해 필터링한다.
- 동일/유사 그룹에서 대표 샘플만 선택해 데이터 다양성을 확보한다.
- split 간(특히 train vs validation/test) 유사 샘플 누수를 방지한다.

## 5) 실행 절차

1. AI Hub 전체 데이터 로드 및 전수 인덱스 생성
2. 각 샘플의 `span`, `complexity` 특징 추출
3. 유사도 기준으로 중복 후보 그룹화 후 필터링
4. 복잡도 구간별 샘플링 비율 적용
5. `train(10k) / validation(1k) / test(500)` 분할
6. 초기 학습 테스트용 `2,000`개를 별도 샘플링/저장
7. 샘플링 완료 후 학습데이터 형태로 변환
8. OCR 정보 주입은 **마지막 단계**에서 수행하고, 대상 split은 `train`, `initial_train_test`로 제한
9. `validation`, `test`는 OCR 미주입(no-OCR) 상태 유지 + AI Hub 원본 기반 HTML 정답(assistant)을 유지
10. OCR 정보는 `text`, `bbox`만 사용하고 `score`는 제외
11. 분할 결과를 인덱스 이력에 기록(재실행 시 중복 방지)

## 5-1) Split별 OCR/정답 정책

- `train`: OCR 주입 (`text`, `bbox`만 유지)
- `initial_train_test`: OCR 주입 (`text`, `bbox`만 유지)
- `validation`: OCR 미주입, AI Hub 원본 기반 HTML 정답 유지
- `test`: OCR 미주입, AI Hub 원본 기반 HTML 정답 유지

## 6) 검증 체크리스트

- split별 목표 샘플 수 충족 여부
- 복잡도 구간별 목표 비율 충족 여부
- split 내부/사이 중복 및 유사 중복 비율 확인
- `sample_id` 기준 중복 선택 0건 확인

## 7) 실행 명령 예시

`train/vlm` 경로에서 실행:

```bash
python -m data.build_sampling_splits \
  --input data/processed/from_server/aihub_training_sampled_ocr_3000.jsonl \
  --output_dir data/experiments/sampling_v1 \
  --history_path data/experiments/sampling_v1/sampled_history.jsonl \
  --train_count 10000 \
  --validation_count 1000 \
  --test_count 500 \
  --initial_test_count 2000 \
  --ratio_high 0.70 \
  --ratio_mid 0.20 \
  --ratio_low 0.10 \
  --max_per_signature 5 \
  --score_priority \
  --prefer_complex_backfill \
  --seed 42
```

생성 파일:
- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`
- `initial_train_test.jsonl`
- `full_index.jsonl`
- `sampled_history.jsonl`
- `sampling_report.json`

참고:
- `sampled_history.jsonl` 기반으로 반복 실행 시 기존 `sample_id` 재선택을 방지함
- 기본 동작에서 OCR 정보는 `text`, `bbox`만 유지하고 `score`는 제거됨
