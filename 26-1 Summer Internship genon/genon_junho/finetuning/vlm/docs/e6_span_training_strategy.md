# E6: Span 테이블 성능 최대화 학습 전략

## 1. 배경 및 문제 분석

### 현재 성능 (best student: e1_sft)

| 메트릭 | Teacher | e1_sft (Student) | 목표 |
|--------|---------|------------------|------|
| TEDS | 0.587 | 0.549 | >0.55 |
| Span F1 | 0.510 | 0.380 | >0.50 |
| Span Recall | - | 0.360 | >0.45 |
| Span Precision | - | - | - |

### 핵심 병목

- **Span Recall = 0.36**: span 셀의 61%를 놓치고 있음
- **Rowspan 정확도(0.61)** < Colspan 정확도(0.71): rowspan 인식이 더 어려움
- TEDS는 0.525 -> 0.549로 개선했지만 **Span F1은 0.397 -> 0.380으로 오히려 하락**
- 학습 데이터 2,700건은 절대적으로 부족 → AIHub 데이터 확장 필요

### 미활용 리소스

1. **AIHub 데이터**: 병합표+콘텐츠 병합표 **206,229건** 존재 (10,000건 샘플링 가능)
2. `data/augment.py`의 `--span-only` 모드 (span 데이터만 선택적 증강)
3. `train/train_qlora.py`의 multi-phase 학습 구조
4. `utils/span_analyzer.py`의 `compute_sampling_weights()` (span 가중 샘플링)

## 2. 전략 개요: 3-Phase 학습 + AIHub 10K 통합

```
Phase 0: 데이터 준비 (AIHub 10K 샘플링 + 병합 + 증강)
    └─ ~2,700건 → ~23,000건 (약 8.5배 확장)
Phase 1: GT 기반 SFT (baseline 확보)
Phase 2: Span 가중 Fine-tuning (complex 80% 집중)
Phase 3: 극한 Span 특화 (complex 90%, 극저 LR)
```

### 데이터 규모 변화

| 단계 | 건수 | 복잡도 분포 |
|------|------|------------|
| 기존 train | ~2,700 | S:16% M:16% C:68% |
| + AIHub 10K | ~12,700 | S:~19% M+C:~81% |
| + span-only 증강 후 | **~23,000** | S:~11% M+C:~89% |
| eval (불변) | ~300 | 기존과 동일 |

## 3. Phase 0: 데이터 준비

### 0-1. AIHub 10,000건 샘플링

기존 `data/sample_aihub_with_ocr.py`를 사용하여 span 비율 80% 이상인 테이블을 10,000건 샘플링한다.

```bash
cd vlm_train/qwen3_vl_tsr
python -m data.sample_aihub_with_ocr \
    --input_dir ./data/extracted/aihub/Training \
    --output ./data/experiments/e6_span/aihub_sampled_10000.jsonl \
    --sample_count 10000 \
    --min_span_ratio 0.80 \
    --prompt_style chandra_table_with_ocr \
    --bbox_scale 1024 --seed 42
```

### 0-2. 기존 데이터와 병합

```bash
cat ./data/experiments/shared/train.jsonl \
    ./data/experiments/e6_span/aihub_sampled_10000.jsonl \
    > ./data/experiments/e6_span/train_merged.jsonl
```

중복 가능성: ~67건 (2,700 x 10,000 / 404,080) → 무시 가능.

### 0-3. Span-only 증강

병합된 데이터에서 span 포함 테이블만 2배 증강한다.

```bash
python -m data.augment \
    --input ./data/experiments/e6_span/train_merged.jsonl \
    --output ./data/experiments/e6_span/train_augmented.jsonl \
    --image_output_dir ./data/experiments/e6_span/augmented_images/ \
    --augment_factor 2 --span-only
```

### 0-4. 평가 프롬프트 정합성

학습 시 `chandra_table_with_ocr` 프롬프트를 사용하며, 평가 시에는 `--prompt_style` 지정으로 자동 처리된다.

## 4. Phase 1: GT 기반 SFT (Baseline)

증강된 ~23,000건 데이터로 baseline 학습. 데이터 8.5배 증가에 맞춰 하이퍼파라미터 조정.

### Config: `config/exp_e6_span.yaml`

| 설정 | 이전 | 변경 후 | 이유 |
|------|------|---------|------|
| train_file | `train_augmented.jsonl` | 동일 | 증강 데이터 |
| num_train_epochs | 30 | **10** | 데이터 증가로 총 step 유사 |
| gradient_accumulation_steps | 1 | **2** | effective batch 8, 학습 안정성 |
| warmup_ratio | 0.03 | **0.05** | 데이터 증가에 맞게 |
| logging_steps | 20 | **50** | step 수 증가 반영 |
| save_steps | 100 | **300** | step 수 증가 반영 |
| eval_steps | 100 | **300** | step 수 증가 반영 |
| per_device_train_batch_size | 4 | 4 | 유지 |
| learning_rate | 2.0e-5 | 2.0e-5 | 유지 |
| lora_r / lora_alpha | 64 / 128 | 64 / 128 | 유지 |
| early_stopping.patience | 3 | 3 | 유지 |

### 실행

```bash
CONFIG=config/exp_e6_span.yaml PHASE=sft bash distill/run_distill.sh
```

### 검증 기준

- TEDS >= 0.58 (baseline 확보 확인)

## 5. Phase 2: Span 가중 Fine-tuning

Phase 1의 best checkpoint에서 이어서, **span 가중 샘플링** + **낮은 LR**로 추가 학습.

### Config: `config/exp_e6_span_phase2.yaml`

| 설정 | 이전 | 변경 후 | 이유 |
|------|------|---------|------|
| resume_from_adapter | `./output/e6_span/student_sft/final` | 동일 | Phase 1 결과 |
| num_train_epochs | 20 | **5** | 데이터 증가, fine-tuning |
| gradient_accumulation_steps | 1 | **2** | Phase 1과 동일 |
| learning_rate | 5.0e-6 | 5.0e-6 | Phase 1의 1/4 유지 |
| warmup_ratio | 0.1 | 0.1 | 유지 |
| span_weighted_sampling | true | true | |
| span_sampling_ratios.simple | 0.10 | **0.05** | simple 비율 축소 |
| span_sampling_ratios.medium | 0.20 | **0.15** | medium 비율 축소 |
| span_sampling_ratios.complex | 0.70 | **0.80** | complex 집중 강화 |
| logging_steps | 20 | **50** | |
| save_steps | 100 | **300** | |
| eval_steps | 100 | **300** | |

### Span 가중 샘플링 동작 원리

`SpanWeightedTrainer`가 HuggingFace Trainer를 동적 서브클래싱하여 `_get_train_sampler()`를 오버라이드한다.

```
실제 데이터 비율:   simple ~11% / medium+complex ~89%
목표 샘플링 비율:   simple 5%  / medium 15% / complex 80%
```

- complex 테이블이 학습 배치의 80%를 차지하도록 가중 샘플링
- `WeightedRandomSampler` (replacement=True) 사용

### 실행

```bash
CONFIG=config/exp_e6_span_phase2.yaml PHASE=sft bash distill/run_distill.sh
```

### 검증 기준

- Span F1 > 0.45
- TEDS > 0.55

## 6. Phase 3: 극한 Span 특화 (신규)

Phase 2 체크포인트에서 이어서, 극히 낮은 LR + complex 90% 집중으로 span 성능 극대화.

### Config: `config/exp_e6_span_phase3.yaml` (신규 생성)

| 설정 | 값 | 비고 |
|------|-----|------|
| resume_from_adapter | `./output/e6_span_phase2/student_sft/final` | Phase 2 결과 |
| num_train_epochs | 3 | 짧은 추가 학습 |
| gradient_accumulation_steps | 2 | |
| learning_rate | **1.0e-6** | Phase 2의 1/5 |
| warmup_ratio | **0.15** | 충분한 warmup |
| span_weighted_sampling | true | |
| span_sampling_ratios.simple | **0.02** | 최소한의 simple |
| span_sampling_ratios.medium | **0.08** | |
| span_sampling_ratios.complex | **0.90** | 극한 complex 집중 |
| early_stopping.patience | 3 | |
| early_stopping.threshold | **0.0005** | 더 민감한 감지 |

### LoRA Resume 동작

Phase 1 → Phase 2 → Phase 3으로 이어지는 연속 학습. 각 Phase에서 `PeftModel.from_pretrained(model, path, is_trainable=True)`로 이전 LoRA 어댑터를 로드하여 가중치를 유지한 채 이어서 학습한다.

### 실행

```bash
CONFIG=config/exp_e6_span_phase3.yaml PHASE=sft bash distill/run_distill.sh
```

### 검증 기준

- Span F1 > 0.50
- Span Recall > 0.45
- TEDS > 0.55

## 7. 평가

```bash
# Phase별 평가 (Phase 1, 2, 3 각각 실행)
bash eval/run_eval.sh \
    --model output/e6_span_phase3/student_sft/final \
    --test_data data/experiments/shared/eval.jsonl \
    --output_dir eval_results/e6_span_phase3 \
    --prompt_style chandra_table_with_ocr \
    --backend vllm --batch_size 16
```

### 목표 메트릭 요약

| Phase | 메트릭 | 목표 | 비고 |
|-------|--------|------|------|
| Phase 1 | TEDS | >0.58 | baseline 확보 |
| Phase 2 | Span F1 | >0.45 | span 개선 시작 |
| Phase 2 | TEDS | >0.55 | 성능 유지 |
| Phase 3 | Span F1 | >0.50 | teacher 수준 도달 |
| Phase 3 | Span Recall | >0.45 | 핵심 병목 해소 |
| Phase 3 | TEDS | >0.55 | 성능 유지 |

### 모니터링 포인트

- 각 Phase 후 **complexity별 TEDS** 확인하여 simple 성능 하락 여부 점검
- span 테이블과 non-span 테이블 별도 메트릭 비교
- Phase 3에서 overfitting 징후 발생 시 조기 중단

## 8. 구현된 코드 변경 사항

### Config 파일 (이번 변경)

| 파일 | 변경 내용 |
|------|----------|
| `config/exp_e6_span.yaml` | epochs 30→10, grad_accum 1→2, warmup 0.03→0.05, steps 조정 |
| `config/exp_e6_span_phase2.yaml` | epochs 20→5, grad_accum 1→2, sampling ratios 강화 |
| `config/exp_e6_span_phase3.yaml` | **신규 생성** — Phase 3 극한 span 특화 config |

### Python 코드 (이전에 구현 완료)

| 파일 | 내용 |
|------|------|
| `distill/student_sft.py` | `SpanWeightedTrainer`, `resume_from_adapter` 지원 |
| `eval/evaluate.py` | `--prompt_style` 인자, `resolve_eval_prompt()` |
| `eval/run_eval.sh` | `--prompt_style` 전달 |

### 기존 파일 (수정 없이 활용)

- `data/sample_aihub_with_ocr.py` — AIHub 데이터 샘플링
- `data/augment.py` — `--span-only` 모드로 증강
- `utils/span_analyzer.py` — `compute_sampling_weights()` 가중치 계산
- `train/train_qlora.py` — `TSRDataset.get_sampling_weights()` 메서드
- `train/collator.py` — Multimodal Collator
- `distill/run_distill.sh` — PHASE=sft 로 실행

## 9. 전체 실행 순서 요약

```bash
cd vlm_train/qwen3_vl_tsr

# ─── Phase 0: 데이터 준비 ───────────────────────────────────────
# 0-1. AIHub 10K 샘플링
python -m data.sample_aihub_with_ocr \
    --input_dir ./data/extracted/aihub/Training \
    --output ./data/experiments/e6_span/aihub_sampled_10000.jsonl \
    --sample_count 10000 --min_span_ratio 0.80 \
    --prompt_style chandra_table_with_ocr --bbox_scale 1024 --seed 42

# 0-2. 병합
cat ./data/experiments/shared/train.jsonl \
    ./data/experiments/e6_span/aihub_sampled_10000.jsonl \
    > ./data/experiments/e6_span/train_merged.jsonl

# 0-3. Span-only 증강
python -m data.augment \
    --input ./data/experiments/e6_span/train_merged.jsonl \
    --output ./data/experiments/e6_span/train_augmented.jsonl \
    --image_output_dir ./data/experiments/e6_span/augmented_images/ \
    --augment_factor 2 --span-only

# ─── Phase 1: GT SFT (Baseline) ────────────────────────────────
CONFIG=config/exp_e6_span.yaml PHASE=sft bash distill/run_distill.sh

# ─── Phase 2: Span 가중 Fine-tuning ────────────────────────────
CONFIG=config/exp_e6_span_phase2.yaml PHASE=sft bash distill/run_distill.sh

# ─── Phase 3: 극한 Span 특화 ───────────────────────────────────
CONFIG=config/exp_e6_span_phase3.yaml PHASE=sft bash distill/run_distill.sh

# ─── 최종 평가 ──────────────────────────────────────────────────
bash eval/run_eval.sh \
    --model output/e6_span_phase3/student_sft/final \
    --test_data data/experiments/shared/eval.jsonl \
    --output_dir eval_results/e6_span_phase3 \
    --prompt_style chandra_table_with_ocr \
    --backend vllm --batch_size 16
```
