# Evaluation/Test Session Summary (2026-02-20)

## 1) 문서 목적

- 이번 세션에서 수행한 평가/테스트 작업을 한 곳에 정리
- 다음 평가 시 동일 조건 재현 가능하도록 명령/경로/주의사항 기록

## 2) 환경/서버 정보

- 학습 서버: `ssh -p 2222 root@192.168.75.174`
- 서빙 서버: `ssh -p 2220 root@192.168.75.173`
- 서빙 접근 포트(외부): `http://192.168.75.173:8010`
- vLLM OpenAI-compatible endpoint:
  - `/v1/chat/completions`
  - `/v1/models`

## 3) 평가 데이터(고정)

- 126 샘플 평가 파일:
  - `train/vlm/data/processed/from_server/eval_used_data/eval_used_data_20260213_141329/eval_localized.jsonl`
- 최근 평가 공통 설정:
  - backend: `api`
  - batch_size: `8`
  - max_new_tokens: `10000`
  - temperature: `0.0`

## 4) 모델 서빙 상태(이번 세션 기준)

- 서빙 서버에서 확인한 모델 목록 (`/v1/models`):
  - base: `qwen3vl8b-e6ckpt600`
  - LoRA alias: `e6ckpt600`
- e6 checkpoint-600 LoRA 파일 경로:
  - `/models/tsr_lora/e6_span_checkpoint-600`

## 5) 수행한 평가 결과 요약

| Model | Endpoint | Samples | Avg TEDS | Avg TEDS-S | Span F1 | Attr Acc | Avg Inf Time (s/sample) |
|---|---|---:|---:|---:|---:|---:|---:|
| e6ckpt300 (self serving) | `http://192.168.75.173:8010/v1/chat/completions` | 126 | 0.4560 | 0.6238 | 0.2419 | 0.3292 | 4.0249 |
| e6ckpt600 (self serving) | `http://192.168.75.173:8010/v1/chat/completions` | 126 | 0.5025 | 0.6472 | 0.3872 | 0.5025 | 6.4458 |
| qwen3_vl_235b (OpenRouter) | `https://openrouter.ai/api/v1/chat/completions` | 126 | 0.7158 | 0.7759 | 0.7479 | 0.8040 | 7.3505 |

핵심 해석:

- `e6ckpt600`은 `e6ckpt300` 대비 정확도 지표가 전반적으로 개선됨.
- `e6ckpt600`은 `e6ckpt300`보다 추론 시간이 증가함.
- 동일 126샘플 기준에서 `qwen3_vl_235b`가 정확도는 가장 높음.

## 6) 결과 파일 위치

- e6ckpt300:
  - `train/vlm/eval_results/local_api_e6ckpt300_126_20260219_221958/metrics.json`
  - `train/vlm/eval_results/local_api_e6ckpt300_126_20260219_221958/predictions.jsonl`
  - `train/vlm/eval_results/local_api_e6ckpt300_126_20260219_221958/report.html`
- qwen3_vl_235b:
  - `train/vlm/eval_results/local_api_qwen3_vl_235b_126_20260219_223709/metrics.json`
  - `train/vlm/eval_results/local_api_qwen3_vl_235b_126_20260219_223709/predictions.jsonl`
  - `train/vlm/eval_results/local_api_qwen3_vl_235b_126_20260219_223709/report.html`
- e6ckpt600 (이번 세션 신규):
  - `train/vlm/eval_results/local_api_e6ckpt600_126_20260220_101300/metrics.json`
  - `train/vlm/eval_results/local_api_e6ckpt600_126_20260220_101300/predictions.jsonl`
  - `train/vlm/eval_results/local_api_e6ckpt600_126_20260220_101300/report.html`

## 7) 로컬 재실행 명령 (권장)

프로젝트 루트에서:

```bash
cd /Users/shkim/_shkim/01.source/tsr_labs/train/vlm

../../test/test_model/.venv/bin/python -m eval.evaluate \
  --model e6ckpt600 \
  --test_data data/processed/from_server/eval_used_data/eval_used_data_20260213_141329/eval_localized.jsonl \
  --output_dir eval_results/local_api_e6ckpt600_126_$(date +%Y%m%d_%H%M%S) \
  --max_samples 126 \
  --backend api \
  --batch_size 8 \
  --max_new_tokens 10000 \
  --temperature 0.0 \
  --api_url http://192.168.75.173:8010/v1/chat/completions \
  --api_model e6ckpt600
```

리포트 생성:

```bash
../../test/test_model/.venv/bin/python -m eval.visualize \
  --predictions <OUTPUT_DIR>/predictions.jsonl \
  --output <OUTPUT_DIR>/report.html
```

## 8) test_table.py 테스트 관련 정리

관련 파일:

- `test/test_model/test_table.py`
- `test/test_model/test_table.sh`
- `test/test_model/model_profiles.json`

현재 프로필 예시:

- `self_serving`: `url=http://192.168.75.173:8010/v1/chat/completions`, `model=""`
- `qwen3_vl_235b`: OpenRouter endpoint/model 지정

중요 변경/동작:

- `model`이 비어 있어도 `/v1/models`에서 자동 탐지 시도
- PNG 입력의 params payload 캐시 저장/재사용 지원
- 기본 실행 예:

```bash
cd /Users/shkim/_shkim/01.source/tsr_labs/test/test_model
TARGET=self_serving bash test_table.sh
```

## 9) 자주 발생한 이슈와 대응

- 이슈: `ModuleNotFoundError: No module named 'editdistance'`
- 대응: 평가는 `test/test_model/.venv` 기반으로 실행하고 필요한 패키지를 venv에 설치

```bash
cd /Users/shkim/_shkim/01.source/tsr_labs
uv pip install --python test/test_model/.venv/bin/python editdistance
```

- 이슈: `Inference model is empty...`
- 대응: `test_table.py`는 모델 자동 탐지 로직이 있으므로 다음 순서로 해결
  - `--model` 직접 지정
  - `--target` 프로필 모델 사용
  - `TABLE_VLM_MODEL` 환경변수 설정
  - 위가 모두 비어 있으면 `/v1/models` 자동 조회

- 이슈: `max_tokens too large` (context length 초과)
- 대응: `max_tokens <= (model_context - input_tokens)` 조건 만족하도록 `max_new_tokens` 조정

## 10) 참고 문서

- 통합 평가 리포트: `train/vlm/docs/EVAL_REPORT_20260218.md`
- 이번 세션 정리(본 문서): `train/vlm/docs/EVAL_TEST_SESSION_SUMMARY_20260220.md`
