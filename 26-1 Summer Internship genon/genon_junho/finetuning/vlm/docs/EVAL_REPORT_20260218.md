# Evaluation Report (2026-02-18)

## 1. 목적

- 요청 모델 평가/비교:
  - `teacher`
  - `student (원본)`
  - `e1`, `e2`, `e3`, `e4`, `e5`
- TEDS 음수 문제(HTML 정규화/파서 실패 추정) 보정 후 재평가 결과 정리

## 2. TEDS 보정 패치

- 패치 파일: `eval/metrics.py` (`TEDSCalculator._normalize_table_html`)
- **패치 v1** (기존):
  - `<table ... </table>` fragment 추출
  - 파서 fallback (`lxml` -> `html.parser`)
  - TEDS 계산값 clamp (`[0, 1]`)
  - `NaN/Inf` 방어
- **패치 v2** (2026-02-18 추가 — HTML 태그 정규화):
  - 원인: teacher 모델이 `<thead>/<tbody>/<th>` 포함 HTML을 생성하나 GT는 `<tr>/<td>` 단순 구조만 사용 → 트리 구조 불일치로 edit distance 폭증 → TEDS=0
  - `<thead>/<tbody>/<tfoot>` wrapper 태그 제거 (자식 노드 유지)
  - `<th>` → `<td>` 변환
  - `<table ...>` → `<table>` (비구조적 속성 제거)
  - `<br>` → 공백 변환

## 3. 평가 지표 설명

### 3.1 TEDS (Tree-Edit-Distance-based Similarity)

- HTML 테이블을 트리 구조로 변환한 뒤, 두 트리 간 편집 거리(노드 삽입/삭제/치환)를 계산
- `TEDS = 1 - (edit_distance / max_tree_size)`, 범위 `[0, 1]`
- **구조와 셀 텍스트 모두** 비교 → 테이블 인식의 종합 성능 지표
- 참고: PubTabNet (IBM) 기준 표준 메트릭

### 3.2 TEDS-S (TEDS-Structure)

- TEDS와 동일한 트리 편집 거리 방식이나 **셀 텍스트를 무시**하고 순수 구조만 비교
- `<td>`, `<tr>` 구조 및 `colspan/rowspan` 속성의 정확도를 반영
- 텍스트 인식(OCR) 오류의 영향을 배제하고 **구조 인식 능력만** 평가할 때 유용

### 3.3 Span F1

- `colspan > 1` 또는 `rowspan > 1`인 **병합 셀(span cell)**에 대한 Precision/Recall/F1
- 예측과 GT의 span cell을 `(row, col, rowspan, colspan)` 튜플로 비교하여 **위치와 크기 모두 일치**해야 TP로 인정
- 병합 셀이 없는 단순 테이블은 양쪽 모두 span=0이므로 F1=1.0 (완벽) 처리

### 3.4 Attribute Accuracy (Attr Acc)

- **위치가 일치하는 span cell** 중에서 `colspan`과 `rowspan` 값이 **모두 정확한** 비율
- Span F1이 "셀을 찾았는가"를 측정한다면, Attr Acc는 "찾은 셀의 span 값이 맞는가"를 측정
- 위치 매칭이 되지 않은 셀은 계산에서 제외됨

### 3.5 Avg Inference Time

- 샘플당 평균 추론 시간 (초). vLLM 배치 추론 시간을 배치 크기로 나눈 값
- teacher는 235B 파라미터로 `batch_size=4`, student 계열은 8B 파라미터로 `batch_size=16`

## 4. 평가 조건

- 실행일: 2026-02-18
- 평가 샘플 수: `126`
- 평가 백엔드: `vLLM`
- 추론 설정:
  - `max_new_tokens=1024`
  - `tensor_parallel_size=4`
  - 배치:
    - 일반 모델 `batch_size=16`
    - teacher `batch_size=4`
- 데이터:
  - OCR on: `data/experiments/shared/eval.jsonl`
  - OCR off: `data/experiments/shared/eval_ocr_off.jsonl`

## 5. 모델별 평가 결과 (통합)

| Model                       | Prompt/Data | Samples | Avg TEDS | Avg TEDS-S |  Span F1 | Attr Acc | Avg Inf Time (s/sample) |
| --------------------------- | ----------- | ------: | -------: | ---------: | -------: | -------: | ----------------------: |
| qwen3_vl_235b (OpenRouter)  | ocr_on      |     126 | 0.715814 |   0.775908 | 0.747921 | 0.803968 |                  7.3505 |
| teacher (qwen3_vl_235b fp8) | ocr_on      |     126 | 0.613110 |   0.758542 | 0.571938 | 0.687235 |                  4.1530 |
| student_base (qwen3_vl_8b)  | ocr_on      |     126 | 0.520175 |   0.679098 | 0.416801 | 0.545871 |                  0.4510 |
| student_base (qwen3_vl_8b)  | ocr_off     |     126 | 0.545485 |   0.690843 | 0.451610 | 0.580263 |                  0.4460 |
| e1_sft                      | ocr_on      |     126 | 0.581183 |   0.690508 | 0.457281 | 0.577976 |                  0.6250 |
| e2_sft                      | ocr_off     |     126 | 0.549612 |   0.688070 | 0.422132 | 0.559656 |                  0.6520 |
| e3_resp                     | ocr_on      |     126 | 0.526359 |   0.688111 | 0.457662 | 0.546230 |                  0.6330 |
| e4_resp                     | ocr_off     |     126 | 0.500150 |   0.690753 | 0.426302 | 0.537412 |                  0.6650 |
| e5_logit_ckpt200            | ocr_on      |     126 | 0.031880 |   0.048010 | 0.125063 | 0.146825 |                  0.6300 |
| e6ckpt300 (self serving)    | ocr_on      |     126 | 0.456019 |   0.623767 | 0.241936 | 0.329167 |                  4.0249 |

## 6. 해석 요약

### 6.1 전체 순위

- **통합 기준 TEDS 최고**: `qwen3_vl_235b (OpenRouter)` → `0.7158`
- **2026-02-18 vLLM 실험군 기준 최고**: `teacher (ocr_on)` → `0.6131`
- **Student 중 최고**: `e1_sft (ocr_on)` → `0.5812` (teacher 대비 95%)
- **TEDS 기준 차선**: `e2_sft (ocr_off)` → `0.5496`
- **student_base (ocr_off)**가 `e3_resp/e4_resp`보다 높으며, SFT 계열(`e1/e2`)만 base를 명확히 상회
- **e5_logit_ckpt200**은 보정 후에도 TEDS `0.032`로 극히 낮음 → 학습 실패

### 6.2 e1/e2 (SFT) > e3/e4 (Response Distillation) 원인 분석

e1/e2는 distillation 없이 GT 데이터로 직접 SFT했고, e3/e4는 teacher 생성 합성 데이터로 SFT했다. e1/e2가 더 높은 원인:

**1) 학습 데이터 품질 차이**

| 항목                          | e1/e2 (GT)               | e3/e4 (Teacher 합성)          |
| ----------------------------- | ------------------------ | ----------------------------- |
| 데이터 파일                   | `shared/train.jsonl`     | `e3_resp_ocr_on/train.jsonl`  |
| 샘플 수                       | 2,702                    | 1,136                         |
| 유효 테이블 (`</table>` 포함) | 2,702/2,702 (100%)       | 921/1,136 (81%)               |
| 평균 응답 길이                | 1,829 chars              | 2,461 chars                   |
| HTML 형식                     | `<table><tr><td>` (간결) | `<thead>/<tbody>/<th>` (장황) |

- **데이터 양**: GT가 2.4배 많음
- **데이터 완결성**: teacher 응답의 19%가 `max_new_tokens=3072`를 초과하여 `</table>` 없이 잘림 → 불완전한 HTML을 학습 타겟으로 사용
- **형식 일관성**: GT는 평가 데이터와 동일한 `<tr>/<td>` 형식이나, teacher 합성 데이터는 `<thead>/<tbody>/<th>` 형식 → 평가 시 형식 불일치

**2) 학습 효율**

- e1은 `batch_size=6`으로 효율적 학습, e3는 `batch_size=1`로 학습 불안정 가능성
- GT 데이터는 간결한 HTML이므로 `max_seq_length=3072` 내에 충분히 수용됨

**3) 결론**

- Response Distillation의 효과가 발현되지 못한 핵심 원인은 **teacher 생성 데이터의 품질 저하** (truncation 19%, 장황한 형식)
- teacher의 `max_new_tokens=3072`가 부족했거나, 생성 후 품질 필터링(`quality_filter`)에서 truncated 응답을 충분히 걸러내지 못함
- 개선 방향: teacher 생성 시 `max_new_tokens` 증가, 또는 생성 프롬프트에서 간결한 HTML 형식 지시

### 6.3 e5 (Logit Distillation) 실패 분석

e5는 e3의 SFT 체크포인트를 초기값으로 하여 offline logit KD를 수행한 모델이다.

**추론 출력 패턴 (126개 샘플)**

| 패턴                           | e3 (SFT base) | e5 ckpt-200 |      e5 ckpt-best |
| ------------------------------ | ------------: | ----------: | ----------------: |
| 정상 `<table>...</table>` 추출 |            82 |           1 |                 - |
| truncated (no `</table>`)      |            44 |         101 |                 - |
| degenerate (반복 태그)         |             0 |          24 |                 - |
| TEDS                           |         0.526 |       0.032 | -0.063 (clamp 전) |

- **e3는 82/126에서 테이블 추출 성공**했으나, e5는 **1/126만 성공** → logit KD가 생성 품질을 파괴
- e5 ckpt-200의 24개 샘플이 `<thead><thead><thead>...` 무한 반복 패턴 (degenerate repetition)
- 나머지 101개는 `<!DOCTYPE html><html><head><script>...` 등 전체 HTML 문서 boilerplate를 생성하다가 `max_new_tokens=1024`에서 잘림
- `checkpoint-best`는 ckpt-200보다 **더 나쁨** (TEDS -0.063) → early stopping이 올바르게 작동하지 않았거나, eval loss가 생성 품질을 반영하지 못함
- `checkpoint-final`은 **존재하지 않음** → 학습이 early stop 또는 비정상 종료

**실패 원인**

1. **SFT 베이스(e3)의 형식 문제**: e3 자체가 117/126에서 full HTML doc 형식을 생성 → logit KD의 teacher logits도 이 형식 기반
2. **top-k logit 근사**: `save_top_k_logits=50`으로 vocab 152,064 중 50개만 저장 → teacher 분포의 극히 일부만 반영, 나머지 토큰의 확률 정보 소실
3. **형식 강화 루프**: teacher logits가 장황한 HTML 형식의 토큰 분포를 담고 있어, KD가 이 형식을 더 강하게 학습 → boilerplate 생성 증가 → 실제 테이블 내용 도달 전 토큰 소진

## 7. 학습시간(추정)

- `trainer_state`에 runtime이 저장되지 않아 체크포인트 timestamp 기반 추정
- `e1_sft_ocr_on`: epoch 약 `3.47h`, 총 약 `9.21h`
- `e2_sft_ocr_off`: epoch 약 `6.14h`, 총 약 `14.53h`
- `e3_resp_ocr_on`: epoch 약 `2.09h`, 총 약 `5.14h`
- `e4_resp_ocr_off`: epoch 약 `2.28h`, 총 약 `5.17h`
- `e5_logit_ocr_on`: 런타임 로그 미저장으로 정밀 집계 불가
  (아카이브 `checkpoint-200 <-> checkpoint-final` 시각 차 약 `4.6h`, rough)

## 8. 결과 파일 위치 (서버)

- `eval_results/model_compare_20260218_e1e5/`
- `eval_results/model_compare_20260218_baselines/`

## 9. 추가 평가 결과 (2026-02-19)

### 9.1 평가 조건

- 평가 샘플 수: `126` (고정)
- 평가 데이터: `data/processed/from_server/eval_used_data/eval_used_data_20260213_141329/eval_localized.jsonl`
- 평가 백엔드: `api` (`eval.evaluate`의 OpenAI-compatible endpoint 호출)
- 공통 추론 설정:
  - `max_new_tokens=1024`
  - `batch_size=8`
  - `temperature=0.0`

### 9.2 모델별 결과

| Model                      | Endpoint                                         | Samples | Avg TEDS | Avg TEDS-S |  Span F1 | Attr Acc | Avg Inf Time (s/sample) |
| -------------------------- | ------------------------------------------------ | ------: | -------: | ---------: | -------: | -------: | ----------------------: |
| e6ckpt300 (self serving)   | `http://192.168.75.173:8010/v1/chat/completions` |     126 | 0.456019 |   0.623767 | 0.241936 | 0.329167 |                  4.0249 |
| qwen3_vl_235b (OpenRouter) | `https://openrouter.ai/api/v1/chat/completions`  |     126 | 0.715814 |   0.775908 | 0.747921 | 0.803968 |                  7.3505 |

### 9.3 해석 요약

- 동일 126샘플 기준에서 `qwen3_vl_235b`가 `e6ckpt300` 대비 모든 핵심 지표(TEDS, TEDS-S, Span F1, Attr Acc)에서 높았다.
- 추론 속도는 `e6ckpt300`가 더 빠르고(`4.02s/sample`), 정확도는 `qwen3_vl_235b`가 더 높았다(`7.35s/sample`).

### 9.4 결과 파일 위치 (로컬)

- `train/vlm/eval_results/local_api_e6ckpt300_126_20260219_221958/metrics.json`
- `train/vlm/eval_results/local_api_e6ckpt300_126_20260219_221958/predictions.jsonl`
- `train/vlm/eval_results/local_api_e6ckpt300_126_20260219_221958/report.html`
- `train/vlm/eval_results/local_api_qwen3_vl_235b_126_20260219_223709/metrics.json`
- `train/vlm/eval_results/local_api_qwen3_vl_235b_126_20260219_223709/predictions.jsonl`
- `train/vlm/eval_results/local_api_qwen3_vl_235b_126_20260219_223709/report.html`
