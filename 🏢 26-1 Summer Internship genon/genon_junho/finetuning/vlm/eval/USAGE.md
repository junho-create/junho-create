# eval/ - 평가 파이프라인 사용법

## 개요

VLM 모델의 테이블 구조 인식 성능을 평가하는 파이프라인.

- **evaluate.py** — 추론 + 메트릭 계산 (vllm / api / transformers 백엔드)
- **metrics.py** — TEDS(APTED), Span F1, GCA/GSA 메트릭
- **visualize.py** — HTML 리포트 생성
- **aggregate_results.py** — 여러 실험 결과 비교 테이블 출력
- **merge_shard_runs.py** — 분할 실행 결과 병합
- **run_eval.sh** — 평가 + 리포트 통합 래퍼

## 빠른 시작

```bash
cd train/vlm

# API 백엔드 (자체 서빙 서버)
python -m eval.evaluate \
    --model my-model \
    --backend api \
    --api_url http://192.168.75.173:8010/v1/chat/completions \
    --test_data data/experiments/e7_gtfilter/test.jsonl \
    --output_dir eval_results/my_experiment \
    --prompt_style chandra_table_with_ocr

# vLLM 백엔드 (로컬 GPU)
python -m eval.evaluate \
    --model output/checkpoint/final \
    --backend vllm \
    --test_data data/experiments/e7_gtfilter/test.jsonl \
    --output_dir eval_results/my_experiment

# transformers 백엔드 (4-bit 양자화)
python -m eval.evaluate \
    --model output/checkpoint/final \
    --backend transformers \
    --test_data data/experiments/e7_gtfilter/test.jsonl \
    --output_dir eval_results/my_experiment
```

사전 생성된 결과(JSONL)끼리 비교 평가:

```bash
python -m eval.evaluate_precomputed \
    --test_data _train_data/20260225_1_3000/data_split/test.jsonl \
    --pred_data _train_data/20260225_1_3000/data/train_ocr_claude.jsonl \
    --output_dir eval_results/precomputed_claude_test100_$(date +%Y%m%d_%H%M%S) \
    --join_key image_path \
    --gt_field gt_html \
    --pred_field gt_html
```

또는 래퍼 스크립트 사용 (평가 + HTML 리포트 자동 생성):

```bash
bash eval/run_eval.sh \
    --model output/checkpoint/final \
    --backend vllm \
    --test_data data/experiments/e7_gtfilter/test.jsonl \
    --output_dir eval_results/my_experiment
```

## 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model` | (필수) | 모델/어댑터 경로 또는 API 모델명 |
| `--test_data` | (필수) | 테스트 JSONL 경로 |
| `--backend` | `transformers` | 추론 백엔드: `vllm`, `api`, `transformers` |
| `--output_dir` | `eval_results/` | 결과 출력 디렉토리 |
| `--prompt_style` | `chandra_table_with_ocr` | 프롬프트 스타일 (`chandra_table_with_ocr` 또는 `chandra_table_without_ocr`) |
| `--max_samples` | 전체 | 최대 평가 샘플 수 |
| `--batch_size` | `8` | vLLM/API 배치 크기 |
| `--max_new_tokens` | `10000` | 최대 생성 토큰 수 |
| `--gt_source` | `aihub` | GT 소스: `aihub`, `dataset`, `auto` |
| `--base_model` | (자동) | LoRA 어댑터의 베이스 모델 경로 |
| `--api_url` | env | API 엔드포인트 URL |
| `--api_model` | `--model` | API 모델명 (미지정 시 --model 사용) |

## 출력 파일

| 파일 | 설명 |
|------|------|
| `predictions.jsonl` | 샘플별 예측 HTML, GT, TEDS, Span F1 등 |
| `metrics.json` | 전체 집계 메트릭 (평균 TEDS, 복잡도별 등) |
| `report.html` | 시각화 HTML 리포트 (run_eval.sh 사용 시) |

## 시각화

```bash
python -m eval.visualize \
    --predictions eval_results/my_experiment/predictions.jsonl \
    --output eval_results/my_experiment/report.html
```

## 결과 집계

여러 실험 결과를 비교 테이블로 출력:

```bash
# eval_results/ 하위 모든 실험 자동 탐색
python -m eval.aggregate_results --results_dir eval_results/

# TEDS 기준 정렬 + JSON 저장
python -m eval.aggregate_results \
    --results_dir eval_results/ \
    --sort_by avg_teds \
    --output eval_results/comparison.json
```
