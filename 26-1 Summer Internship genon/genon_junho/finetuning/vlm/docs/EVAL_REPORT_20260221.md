# Evaluation Report (2026-02-21)

## 1. 목적/범위

- 2026-02-21 기준으로 누적된 평가 결과를 한 문서로 통합 정리한다.
- 본 문서는 최근 운영 평가(로컬/서빙 API 기반) 중심으로 정리한다.
- 2026-02-18 vLLM 실험군(`teacher`, `student_base`, `e1~e5`) 상세는 기존 문서 `train/vlm/docs/EVAL_REPORT_20260218.md`를 기준으로 한다.

## 2. 비교 시 주의사항

- 평가 세트가 2종이다.
  - 고정 126세트: `train/vlm/data/processed/from_server/eval_used_data/eval_used_data_20260213_141329/eval_localized.jsonl`
  - 500세트: `train/vlm/data/processed/from_server/sampling_prod_lateocr_20260220_171533/final_splits_with_ocr/test.jsonl`
- `max_new_tokens`가 혼재한다.
  - 과거 일부 126평가: `1024`
  - 2026-02-20 이후 표준: `10000`
- 따라서 절대 비교는 같은 `samples + max_new_tokens` 조건끼리 우선 권장한다.

## 3. 126 샘플 누적 결과

| Run | Model | Samples | Backend | max_new_tokens | Avg TEDS | Avg TEDS-S | Span F1 | Attr Acc | Avg Inf Time (s/sample) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| `local_api_e6ckpt300_126_20260219_221958` | e6ckpt300 (self serving) | 126 | api | 1024 | 0.4560 | 0.6238 | 0.2419 | 0.3292 | 4.0249 |
| `local_api_e6ckpt600_126_20260220_101300` | e6ckpt600 (self serving) | 126 | api | 1024 | 0.5025 | 0.6472 | 0.3872 | 0.5025 | 6.4458 |
| `local_api_qwen3_vl_235b_126_20260219_223709` | qwen3_vl_235b (OpenRouter, local) | 126 | api | 1024 | 0.7158 | 0.7759 | 0.7479 | 0.8040 | 7.3505 |
| `local_api_qwen3_vl_235b_126_max10000_20260220_155707` | qwen3_vl_235b (OpenRouter, local) | 126 | api | 10000 | 0.8101 | 0.9029 | 0.7469 | 0.8317 | 15.1149 |
| `local_api_anthropic_claude_opus_4_6_126_max10000_20260220_165822` | anthropic/claude-opus-4.6 (OpenRouter, local) | 126 | api | 10000 | 0.7671 | 0.8903 | 0.6943 | 0.7228 | 6.9451 |
| `serving_api_qwen_qwen3_vl_235b_a22b_instruct_126_20260220_010529` | qwen/qwen3-vl-235b-a22b-instruct (serving) | 126 | api | 1024 | 0.5840 | 0.6974 | 0.6163 | 0.6806 | 7.1366 |
| `serving_api_qwen_qwen3_vl_8b_instruct_126_20260220_012034` | qwen/qwen3-vl-8b-instruct (serving) | 126 | api | 1024 | 0.5050 | 0.6397 | 0.4579 | 0.5308 | 2.2120 |
| `serving_api_qwen_qwen3_vl_235b_a22b_instruct_126_max10000_20260220_014052` | qwen/qwen3-vl-235b-a22b-instruct (serving) | 126 | api | 10000 | 0.6940 | 0.8789 | 0.6064 | 0.6690 | 22.9898 |
| `serving_api_qwen_qwen3_vl_8b_instruct_126_max10000_20260220_022916` | qwen/qwen3-vl-8b-instruct (serving) | 126 | api | 10000 | 0.6209 | 0.8415 | 0.4595 | 0.5302 | 8.9500 |
| `serving_api_anthropic_claude_opus_4_6_126_max10000_20260220_024810` | anthropic/claude-opus-4.6 (serving) | 126 | api | 10000 | 0.7894 | 0.9115 | 0.7174 | 0.7626 | 6.9469 |
| `serving_api_moonshotai_kimi_k2_5_126_max10000_20260220_030251` | moonshotai/kimi-k2.5 (serving) | 126 | api | 10000 | 0.6208 | 0.7109 | 0.6706 | 0.7232 | 25.0441 |

## 4. 500 샘플 종합 결과 (`sampling_prod_lateocr_20260220_171533`)

| Run | Model | Samples | Avg TEDS | Avg TEDS-S | Span F1 | Attr Acc | Avg Inf Time (s/sample) |
|---|---|---:|---:|---:|---:|---:|---:|
| `serving_api_e1init2000_sampling_prod_lateocr_20260220_171533_500_max10000_20260220_224807` | e1init2000 (self serving + LoRA) | 500 | 0.6726 | 0.8350 | 0.5145 | 0.6074 | 3.2202 |
| `serving_api_qwen_qwen3_vl_235b_a22b_instruct_sampling_prod_lateocr_20260220_171533_500_max10000_20260220_194623` | qwen/qwen3-vl-235b-a22b-instruct | 500 | 0.6865 | 0.8448 | 0.5457 | 0.6451 | 8.3535 |
| `serving_api_qwen_qwen3_vl_8b_instruct_sampling_prod_lateocr_20260220_171533_500_max10000_20260220_205612` | qwen/qwen3-vl-8b-instruct | 500 | 0.6282 | 0.8075 | 0.4793 | 0.5816 | 4.2718 |
| `serving_api_anthropic_claude_opus_4_6_sampling_prod_lateocr_20260220_171533_500_max10000_20260220_213200` | anthropic/claude-opus-4.6 | 500 | 0.7641 | 0.8864 | 0.6978 | 0.7560 | 3.7385 |
| `serving_api_moonshotai_kimi_k2_5_sampling_prod_lateocr_20260220_171533_500_max10000_20260220_220321` | moonshotai/kimi-k2.5 | 500 | 0.6516 | 0.7427 | 0.6187 | 0.6736 | 26.1480 |

## 5. 500 샘플 복잡도별 성능

### simple

| Model | n | TEDS | TEDS-S | Span F1 | Attr Acc | Avg Inf Time (s/sample) |
|---|---:|---:|---:|---:|---:|---:|
| e1init2000 | 50 | 0.7283 | 0.9046 | 0.7800 | 0.7800 | 3.5853 |
| qwen/qwen3-vl-235b-a22b-instruct | 50 | 0.8722 | 0.9496 | 0.8600 | 0.8600 | 9.1781 |
| qwen/qwen3-vl-8b-instruct | 50 | 0.7229 | 0.8954 | 0.6600 | 0.6600 | 4.2350 |
| anthropic/claude-opus-4.6 | 50 | 0.9319 | 0.9811 | 0.9800 | 0.9800 | 4.0548 |
| moonshotai/kimi-k2.5 | 50 | 0.7887 | 0.8354 | 0.9200 | 0.9200 | 26.2243 |

### medium

| Model | n | TEDS | TEDS-S | Span F1 | Attr Acc | Avg Inf Time (s/sample) |
|---|---:|---:|---:|---:|---:|---:|
| e1init2000 | 100 | 0.6336 | 0.7924 | 0.3075 | 0.3350 | 2.8641 |
| qwen/qwen3-vl-235b-a22b-instruct | 100 | 0.6569 | 0.8213 | 0.4077 | 0.4300 | 7.3374 |
| qwen/qwen3-vl-8b-instruct | 100 | 0.5367 | 0.7318 | 0.3094 | 0.3800 | 3.6334 |
| anthropic/claude-opus-4.6 | 100 | 0.7023 | 0.8410 | 0.4721 | 0.4900 | 3.4528 |
| moonshotai/kimi-k2.5 | 100 | 0.6385 | 0.7688 | 0.4393 | 0.4600 | 25.5177 |

### complex

| Model | n | TEDS | TEDS-S | Span F1 | Attr Acc | Avg Inf Time (s/sample) |
|---|---:|---:|---:|---:|---:|---:|
| e1init2000 | 350 | 0.6758 | 0.8373 | 0.5357 | 0.6606 | 3.2698 |
| qwen/qwen3-vl-235b-a22b-instruct | 350 | 0.6685 | 0.8366 | 0.5402 | 0.6758 | 8.5260 |
| qwen/qwen3-vl-8b-instruct | 350 | 0.6408 | 0.8165 | 0.5021 | 0.6280 | 4.4594 |
| anthropic/claude-opus-4.6 | 350 | 0.7577 | 0.8858 | 0.7220 | 0.8000 | 3.7749 |
| moonshotai/kimi-k2.5 | 350 | 0.6357 | 0.7221 | 0.6270 | 0.6995 | 26.3172 |

## 6. 핵심 요약

- 500샘플 기준 정확도 1위는 `anthropic/claude-opus-4.6`였다.
  - `TEDS=0.7641`, `TEDS-S=0.8864`, `Span F1=0.6978`, `Attr Acc=0.7560`
- 500샘플 기준 추론 속도(짧을수록 빠름) 1위는 `e1init2000`이었다.
  - `3.2202s/sample`
- `qwen/qwen3-vl-235b-a22b-instruct`는 `e1init2000`보다 정확도는 높고, 속도는 느렸다.
- `moonshotai/kimi-k2.5`는 Span/Attr 지표는 준수하나 평균 추론 시간이 매우 길어 운영 비용/지연 측면에서 불리했다.

## 7. 관련 아티팩트

- 종합 지표 원본: 각 run 디렉터리의 `metrics.json`, `predictions.jsonl`
  - 경로 루트: `train/vlm/eval_results/`
- 500 복잡도 요약 TSV:
  - `train/vlm/eval_results/openrouter_500_complexity_summary_20260221.tsv`
- e1 학습률 로그:
  - 이미지: `train/vlm/output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/learning_rate_log.png`
  - 원본값(CSV): `train/vlm/output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/learning_rate_log.csv`
- e1 학습 진행 주요 로그(손실/grad/LR/평가속도):
  - 통합 로그 이미지: `train/vlm/output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/training_key_logs.png`
  - loss 곡선 이미지: `train/vlm/output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/training_loss_curves.png`
  - eval runtime/throughput 이미지: `train/vlm/output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/eval_runtime_throughput.png`
  - 로그 CSV: `train/vlm/output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/training_key_logs.csv`
  - 요약 JSON/CSV:
    - `train/vlm/output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/training_key_logs_summary.json`
    - `train/vlm/output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/training_key_logs_summary.csv`
- 기존 상세(2026-02-18 실험군):
  - `train/vlm/docs/EVAL_REPORT_20260218.md`

![e1 training key logs](../output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/training_key_logs.png)

![e1 training loss curves](../output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/training_loss_curves.png)

![e1 eval runtime throughput](../output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/eval_runtime_throughput.png)

![e1 learning rate log](../output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/learning_rate_log.png)
