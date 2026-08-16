# e1 학습 요약 (2026-02-21)

## 1) 개요

- 목적: 기존 `e1` 학습 설정을 유지하고, 데이터셋만 교체하여 SFT 재학습
- 학습 데이터: `initial_train_test` 2,000건
- 검증 데이터: `validation` 1,000건
- 결과: `checkpoint-300`이 best, 최종 `final` 체크포인트 생성 완료

## 2) 실행 정보

- 실행 서버: `ssh -p 2222 root@192.168.75.174`
- 실행 시각(KST):
  - 시작: `2026-02-20 18:45:34`
  - 종료: `2026-02-21 13:57:08`
- 총 소요시간: `19:11:34` (로그 기준 약 `69,100s`)
- 실행 설정 파일:
  - `/home/vlm_train/qwen3_vl_tsr/config/exp_e1_initial_test2000_val1000.yaml`
- 실행 명령(요약):
  - `CONFIG=config/exp_e1_initial_test2000_val1000.yaml PHASE=sft NUM_GPUS=4 MASTER_PORT=29620 bash distill/run_distill.sh`

## 3) 학습 설정

- Base model: `Qwen/Qwen3-VL-8B-Instruct`
- 방식: Student SFT (Output Distillation 파이프라인의 SFT 단계)
- GPU: `4` (`CUDA_VISIBLE_DEVICES=0,1,2,3`)
- Epoch 설정: `50`
- LR: `2e-05`
- Early stopping:
  - metric: `eval_loss`
  - patience: `2`
  - threshold: `0.001`
- Trainable params:
  - `174,587,904 / 8,941,711,600` (`1.9525%`)

### LoRA 주요 옵션

학습 config 및 최종 adapter 설정 기준:

- 참조 파일:
  - 학습 config: `/home/vlm_train/qwen3_vl_tsr/config/exp_e1_initial_test2000_val1000.yaml`
  - 최종 adapter: `train/vlm/output/from_server/e1_initial_test2000_val1000_full_20260221_1/final/adapter_config.json`

| 항목 | 값 |
|---|---|
| `use_lora` | `true` |
| `lora_r` (`r`) | `64` |
| `lora_alpha` | `128` |
| `lora_dropout` | `0.05` |
| `bias` | `none` |
| `peft_type` | `LORA` |
| `task_type` | `CAUSAL_LM` |
| `target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |

## 4) 데이터셋 정보

- Train (`2000`):
  - `/home/vlm_train/qwen3_vl_tsr/data/processed/from_server/sampling_prod_lateocr_20260220_171533/final_splits_with_ocr/initial_train_test.jsonl`
- Validation (`1000`):
  - `/home/vlm_train/qwen3_vl_tsr/data/processed/from_server/sampling_prod_lateocr_20260220_171533/final_splits_with_ocr/validation.jsonl`
- 참고:
  - split 정책상 `train/initial_train_test`는 OCR 적용, `validation/test`는 no-OCR 유지

## 5) 학습 결과

- 출력 경로:
  - `/home/vlm_train/qwen3_vl_tsr/output/e1_sft_ocr_on_initial_test2000_val1000/student_sft`
- 생성 체크포인트:
  - `checkpoint-300`
  - `checkpoint-500`
  - `final`
- 최적 체크포인트:
  - `checkpoint-300`
  - `best eval_loss = 0.050603095442056656`
- 종료 상태:
  - `global_step=500 / max_steps=4200`
  - `epoch=5.9523809523809526`
  - 조기 종료 조건 충족 후 종료

### Eval loss 추이 (trainer_state 기준)

| step | epoch | eval_loss |
|---:|---:|---:|
| 100 | 1.1905 | 0.0817487016 |
| 200 | 2.3810 | 0.0547125302 |
| 300 | 3.5714 | 0.0506030954 |
| 400 | 4.7619 | 0.0525672771 |
| 500 | 5.9524 | 0.0563441254 |

## 6) 로그/산출물

- 서버 학습 로그:
  - `/home/vlm_train/qwen3_vl_tsr/output/e1_sft_ocr_on_initial_test2000_val1000/tmux_train_20260220_184534.log`
- 로컬 동기화 경로:
  - `train/vlm/output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/`
- 로컬 보관 파일:
  - `train/vlm/output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/tmux_train_20260220_184534.log`
  - `train/vlm/output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/trainer_state.json`
  - `train/vlm/output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/training_progress_downloaded.html`
  - `train/vlm/output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/training_progress_downloaded.json`
  - `train/vlm/output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/training_key_logs.csv`
  - `train/vlm/output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/training_key_logs_summary.json`
  - `train/vlm/output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/training_key_logs_summary.csv`
  - `train/vlm/output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/training_key_logs.png`
  - `train/vlm/output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/training_loss_curves.png`
  - `train/vlm/output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/eval_runtime_throughput.png`
  - `train/vlm/output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/learning_rate_log.png`
  - `train/vlm/output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/learning_rate_log.csv`

### 학습 진행 핵심 로그 요약

| 항목 | 값 |
|---|---:|
| Train log points | 25 |
| Eval log points | 5 |
| Train loss (min) | `0.0233 @ step 460` |
| Train loss (last) | `0.0283 @ step 500` |
| Eval loss (min / best) | `0.0506030954 @ step 300` |
| Eval loss (last) | `0.0563441254 @ step 500` |
| Grad norm (max) | `1.2675627470 @ step 20` |
| Grad norm (last) | `0.3956721425 @ step 500` |
| Learning rate (min) | `3.015873e-06 @ step 20` |
| Learning rate (max) | `1.999950e-05 @ step 140` |
| Eval runtime (avg) | `4528.5612 s` |
| Eval samples/sec (avg) | `0.2208` |
| Eval steps/sec (avg) | `0.0550` |
| Train runtime (summary) | `69060.3422 s` |
| Train samples/sec (summary) | `1.448` |
| Train steps/sec (summary) | `0.061` |
| Train loss (summary) | `0.0711169299` |

### 학습 진행 통합 로그(LOSS/LR/GradNorm/Eval Loss)

![e1 training key logs](../output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/training_key_logs.png)

### Loss 곡선 (Train vs Eval)

![e1 training loss curves](../output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/training_loss_curves.png)

### Eval Runtime / Throughput 로그

![e1 eval runtime throughput](../output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/eval_runtime_throughput.png)

### 학습률 로그 (단일)

![e1 learning rate log](../output/from_server/e1_sft_ocr_on_initial_test2000_val1000_20260221/learning_rate_log.png)

## 7) 후속 서빙/평가 결과 (동일 세션)

- 서빙 서버: `ssh -p 2220 root@192.168.75.173`
- vLLM endpoint: `http://127.0.0.1:8011`
- 서빙 모델 ID: `e1init2000` (`base: e1init2000-base`)
- 500샘플 평가 결과 (`gt_source=dataset`):
  - output:
    - `train/vlm/eval_results/serving_api_e1init2000_sampling_prod_lateocr_20260220_171533_500_max10000_20260220_224807`
  - 주요 지표:
    - `avg_teds=0.6726221666`
    - `avg_teds_structure=0.8350319084`
    - `avg_span_f1=0.5145069637`
    - `avg_attribute_accuracy=0.6074230159`
    - `avg_inference_time=3.2201804676 s/sample`

## 8) 해석 요약

- 데이터셋 교체 조건(`train=2000`, `validation=1000`)으로 e1 재학습이 정상 완료됨.
- `step 300` 이후 검증 손실이 반등하여 early stopping으로 `step 500`에서 종료됨.
- 학습 완료 모델은 서빙/500샘플 평가까지 연계 확인되었고, 실사용 가능한 결과물이 생성됨.
