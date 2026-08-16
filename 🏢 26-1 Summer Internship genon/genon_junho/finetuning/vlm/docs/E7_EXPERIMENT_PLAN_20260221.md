# E7 실험 계획: 데이터 스케일업 + Medium 강화 2-Phase 학습

## 1. 배경

### 1.1 e1init2000 학습 결과 (선행 실험)

e1init2000은 Qwen3-VL-8B-Instruct 기반 LoRA SFT 모델로, AIHub GT 2,000건으로 학습되었다.
500샘플 평가에서 8B 모델 중 최고 성능이자 가장 빠른 추론 속도를 달성했다.

| Metric | e1init2000 | 235B Teacher | claude-opus-4.6 |
|--------|-----------|-------------|-----------------|
| TEDS | 0.6726 | 0.6865 | **0.7641** |
| TEDS-S | 0.8350 | 0.8448 | **0.8864** |
| Span F1 | 0.5145 | 0.5457 | **0.6978** |
| Attr Acc | 0.6074 | 0.6451 | **0.7560** |
| 추론 속도 (s/sample) | **3.22** | 8.35 | 3.74 |

- 8B 모델이 235B Teacher TEDS의 98%에 도달 (0.6726 vs 0.6865)
- TEDS-S=0.835로 테이블 구조 인식은 견실
- claude-opus-4.6 (TEDS=0.7641) 대비 격차 존재

### 1.2 복잡도별 성능 (e1init2000, 500샘플)

| 복잡도 | n | TEDS | Span F1 | Attr Acc |
|--------|--:|-----:|--------:|---------:|
| Simple | 50 | 0.7283 | 0.7800 | 0.7800 |
| **Medium** | **100** | **0.6336** | **0.3075** | **0.3350** |
| Complex | 350 | 0.6758 | 0.5357 | 0.6606 |

**Medium 복잡도에서 Span F1=0.31로 심각한 약점** — Simple(0.78), Complex(0.54) 대비 최악.

---

## 2. 학습 진단

### 2.1 핵심 문제점

| 문제 | 증거 | 원인 |
|------|------|------|
| **과적합** | train loss 하락(0.089→0.028) but eval loss step 300에서 반등(0.0506→0.0563) | 2,000건으로 ~6 epoch 반복 → 데이터 암기 |
| **Medium 취약** | Medium Span F1=0.31 (최악) | 학습 데이터의 medium 비율 부족 + "경계 사례" 난이도 |
| **Span 값 오류** | Position F1(0.62) > Span F1(0.51) = 0.10 격차 | 병합 위치 감지 OK, colspan/rowspan 정확한 값 결정에 실패 |

### 2.2 학습 효율 분석

```
Steps/Epoch = 2000 / (6 batch × 4 GPU) ≈ 84 steps
Best: step 300 = epoch 3.57 → sweet spot 약 3~4 epoch
Step 500 = epoch 5.95 → 조기 종료
```

2,000건 데이터에서 3~4 epoch이 최적. 그 이후는 과적합 구간.

### 2.3 핵심 교훈

- **GT 데이터 품질 > 합성 데이터 양** (e1init2000 GT 2K > E6 합성 23K)
- SFT(e1/e2) > Response Distill(e3/e4) > Logit KD(e5, 실패)
- Early stopping이 적시에 동작하여 과적합 확대 방지

---

## 3. E7 개선 전략

**접근: "데이터 스케일업 + Medium 비중 강화 + 2-Phase 학습"**

AIHub GT 데이터를 10K로 스케일업하되, medium 복잡도 비중을 강화한다.

### 복잡도 분포 변화

| 복잡도 | e1init2000 (2K) | E7 (10K) | 배수 |
|--------|----------------|---------|-----:|
| Simple | ~200건 (10%) | ~1,500건 (15%) | 7.5x |
| **Medium** | ~400건 (20%) | **~3,500건 (35%)** | **8.75x** |
| Complex | ~1,400건 (70%) | ~5,000건 (50%) | 3.6x |

---

## 4. 실행 계획

### Phase 0: 데이터 준비

#### 0-1. AIHub 15K 샘플링

```bash
cd /home/vlm_train/qwen3_vl_tsr

python -m data.sample_aihub_with_ocr \
    --input_dir ./data/extracted/aihub/Training \
    --output ./data/processed/aihub_15k_medium_focus.jsonl \
    --sample_count 15000 \
    --selection_mode full_analysis \
    --min_span_ratio 0.60 \
    --prompt_style chandra_table_with_ocr \
    --bbox_scale 1024 --seed 2026
```

- `full_analysis`: 전체 HTML 복잡도 분석 후 난이도 우선 샘플링
- `min_span_ratio 0.60`: medium+complex 비율 최소 60% 보장

#### 0-2. Split 구성

```bash
python -m data.build_sampling_splits \
    --input ./data/processed/aihub_15k_medium_focus.jsonl \
    --output_dir ./data/experiments/e7_scale \
    --train_count 10000 --validation_count 1000 \
    --test_count 500 --initial_test_count 2000 \
    --ratio_high 0.50 --ratio_mid 0.35 --ratio_low 0.15 \
    --max_per_signature 3 --score_priority --prefer_complex_backfill \
    --seed 2026
```

**출력 파일:**
- `e7_scale/train.jsonl` (10,000건)
- `e7_scale/validation.jsonl` (1,000건)
- `e7_scale/test.jsonl` (500건)

### Phase 1: 구조 기초 학습

**Config: `config/exp_e7_phase1.yaml`**

| 항목 | e1init2000 | E7 Phase 1 | 변경 이유 |
|------|-----------|------------|----------|
| train_file | initial_train_test (2K) | e7_scale/train (10K) | 5x 스케일업 |
| eval_file | validation (1K) | e7_scale/validation (1K) | 동일 |
| num_train_epochs | 50 | **5** | 데이터 증가, 과적합 방지 |
| per_device_train_batch_size | 6 | **4** | 더 많은 steps/epoch |
| gradient_accumulation_steps | 1 | **2** | effective batch=32, 안정성 향상 |
| learning_rate | 2e-5 | 2e-5 | 유지 |
| warmup_ratio | 0.03 | **0.05** | 데이터 증가에 맞게 |
| max_seq_length | 3072 | **4096** | 큰 테이블 HTML 잘림 방지 |
| save_steps / eval_steps | 100 | **500** | steps 증가 반영 |
| early_stopping.patience | 2 | **3** | 데이터 많아지면 변동 완만 |
| max_train_samples | 3000 | **(제거)** | 전체 10K 사용 |

**Steps 예상:**
```
Steps/Epoch = 10000 / (4 batch × 4 GPU × 2 accum) = 10000 / 32 ≈ 313
Total steps ≈ 313 × 5 = 1,563
Eval 횟수 ≈ 3회 (step 500, 1000, 1500)
```

**실행:**
```bash
CONFIG=config/exp_e7_phase1.yaml PHASE=sft NUM_GPUS=4 MASTER_PORT=29620 \
    bash distill/run_distill.sh
```

### Phase 2: Span 특화 Fine-tuning

Phase 1 best checkpoint에서 이어서, medium+complex 가중 샘플링으로 Span 성능 집중 개선.

**Config: `config/exp_e7_phase2.yaml`**

| 항목 | Phase 1 | Phase 2 | 변경 이유 |
|------|---------|---------|----------|
| resume_from_adapter | (없음) | `./output/e7_phase1/student_sft/final` | Phase 1 이어서 |
| num_train_epochs | 5 | **3** | 짧은 추가 학습 |
| learning_rate | 2e-5 | **5e-6** | Phase 1의 1/4 |
| warmup_ratio | 0.05 | **0.10** | 짧은 학습에 충분한 warmup |
| span_weighted_sampling | (비활성) | **true** | 가중 샘플링 활성화 |
| span_sampling_ratios | - | simple=0.05, **medium=0.45**, complex=0.50 | Medium 집중 |

**실행:**
```bash
CONFIG=config/exp_e7_phase2.yaml PHASE=sft NUM_GPUS=4 MASTER_PORT=29620 \
    bash distill/run_distill.sh
```

---

## 5. LoRA 설정 (Phase 1/2 공통)

| 항목 | 값 | 소스 | Config 변경 가능 |
|------|-----|------|:---:|
| `lora_r` | 64 | config YAML | O |
| `lora_alpha` | 128 | config YAML | O |
| `lora_dropout` | 0.05 | `student_sft.py:217` 하드코딩 | X |
| `bias` | "none" | `student_sft.py:218` 하드코딩 | X |
| `task_type` | "CAUSAL_LM" | `student_sft.py:219` 하드코딩 | X |
| `target_modules` | q/k/v/o_proj, gate/up/down_proj | `student_sft.py:220-222` 하드코딩 | X |

- alpha/r ratio = 2.0 (표준 설정)
- E1, E6 실험과 동일한 LoRA 설정 → 실험 간 비교 가능
- r=64 유지 이유: e1init2000에서 충분히 검증됨, r=128 시 OOM 위험

### Phase 2 resume_from_adapter 동작

Phase 2에서는 `PeftModel.from_pretrained()`로 Phase 1 adapter를 로드한다 (`student_sft.py:201-212`).
이때 config YAML의 `lora_r/lora_alpha`는 무시되고 Phase 1의 `adapter_config.json`이 우선한다.
Phase 1과 Phase 2 설정이 동일(r=64, alpha=128)하므로 문제 없음.

---

## 6. 기대 성능 목표

| Metric | e1init2000 | E7 목표 | opus 참조 |
|--------|-----------|---------|----------|
| TEDS | 0.6726 | **0.73~0.76** | 0.7641 |
| TEDS-S | 0.8350 | **0.87~0.89** | 0.8864 |
| Span F1 | 0.5145 | **0.60~0.65** | 0.6978 |
| Medium Span F1 | 0.3075 | **0.45~0.50** | 0.4721 |
| Attr Acc | 0.6074 | **0.68~0.72** | 0.7560 |

---

## 7. 검증 방법

### 7.1 학습 모니터링

- eval_loss가 Phase 1에서 안정적으로 하강하는지 확인
- train loss vs eval loss 격차(과적합 갭) 모니터링

### 7.2 평가

```bash
# Phase 1/2 완료 후 서빙 + 500샘플 평가
# 동일 test.jsonl, max_new_tokens=10000, gt_source=dataset
bash eval/run_eval.sh \
    --backend api \
    --api_url http://127.0.0.1:8011/v1/chat/completions \
    --api_model e7_phase2 \
    --test_data ./data/experiments/e7_scale/test.jsonl \
    --max_samples 500 \
    --output_dir eval_results/e7_phase2_500
```

### 7.3 성공 기준

- **Phase 1**: TEDS >= 0.70 (baseline 확보)
- **Phase 2**: Span F1 >= 0.58, Medium Span F1 >= 0.40

---

## 8. 피해야 할 것 (실패 교훈)

| 금지 사항 | 실패 근거 |
|-----------|----------|
| Logit KD | E5에서 TEDS=0.032로 완전 실패 (top-k=50 근사 + 형식 파괴) |
| 합성 데이터 과의존 | E6 합성 23K < e1init2000 GT 2K 성능 |
| Response Distillation 단독 | E3/E4 base 수준 (teacher 합성 데이터 19% truncation) |
| `max_train_samples` + `span_weighted_sampling` 동시 사용 | Subset 생성 시 가중 샘플링 비활성화됨 (`student_sft.py:496-500` 경고) |

---

## 9. 관련 파일

### Config 파일

| 파일 | 용도 |
|------|------|
| `config/exp_e7_phase1.yaml` | Phase 1 구조 기초 학습 설정 |
| `config/exp_e7_phase2.yaml` | Phase 2 Span 특화 학습 설정 |

### 참조 코드 (수정 없음)

| 파일 | 역할 | 핵심 기능 |
|------|------|----------|
| `distill/student_sft.py` | SFT 학습기 | `resume_from_adapter` (L201-212), `span_weighted_sampling` (L487-501) |
| `data/sample_aihub_with_ocr.py` | AIHub 샘플링 | `full_analysis`, `min_span_ratio` |
| `data/build_sampling_splits.py` | Split 구성 | `ratio_high/mid/low`, `max_per_signature`, `score_priority` |
| `eval/evaluate.py` | 평가 | TEDS, Span F1, Attr Acc |

### 선행 문서

| 문서 | 내용 |
|------|------|
| `docs/E1_TRAINING_SUMMARY_20260221.md` | e1init2000 학습 상세 |
| `docs/EVAL_REPORT_20260221.md` | 500샘플 평가 결과 통합 |
| `docs/e6_span_training_strategy.md` | E6 3-Phase 전략 (참고) |

---

## 10. 전체 실행 순서 요약

```bash
# 서버: ssh -p 2222 root@192.168.75.174
cd /home/vlm_train/qwen3_vl_tsr

# Phase 0: 데이터 준비 (수 시간)
python -m data.sample_aihub_with_ocr ...   # → aihub_15k_medium_focus.jsonl
python -m data.build_sampling_splits ...    # → e7_scale/{train,validation,test}.jsonl

# Phase 1: 구조 기초 학습 (~24h)
CONFIG=config/exp_e7_phase1.yaml PHASE=sft NUM_GPUS=4 MASTER_PORT=29620 \
    bash distill/run_distill.sh

# Phase 2: Span 특화 학습 (~8h)
CONFIG=config/exp_e7_phase2.yaml PHASE=sft NUM_GPUS=4 MASTER_PORT=29620 \
    bash distill/run_distill.sh

# 평가: 서빙 서버에 모델 배포 후 500샘플 평가
bash eval/run_eval.sh ...
```
