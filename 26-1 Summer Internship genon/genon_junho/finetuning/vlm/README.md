# Qwen3-VL TSR Toolkit

Qwen3-VL 기반 Table Structure Recognition(TSR) 학습 및 지식 증류 도구.
Teacher(Qwen3-VL-235B-A22B) 모델의 지식을 Student(Qwen3-VL-8B) 모델로 증류하여,
적은 GPU 자원으로도 높은 테이블 인식 성능을 달성하는 것이 목표다.

**진입점**
- 증류 파이프라인: `distill/run_distill.sh`
- 독립 학습(QLoRA): `train/run_train.sh`

```text
qwen3_vl_tsr/
├── config/     # 학습/증류 YAML 설정 파일 (7종)
├── data/       # 데이터 변환, 샘플링, 분할 스크립트
├── distill/    # 증류 파이프라인 (generate, filter, SFT, logit, feature)
├── train/      # 독립 QLoRA 학습 코드
├── eval/       # 평가 및 시각화
└── utils/      # HTML 파싱, 프롬프트 템플릿, span 분석
```

---

## 아키텍처

```text
AIHub ZIP
  │
  ▼
Phase 0: Prepare (압축 해제)
  │
  ├─ Path A ─── sample_aihub_with_ocr ──► split_dataset ──► SFT ──► Eval
  │              (3000개 샘플링+OCR)
  │
  └─ Path B ─── Phase 1: Generate ──► Phase 2: Filter ──► Phase 3: SFT ──► Eval
                 (Teacher 합성)         (품질 필터링)       (Student 학습)
                                                               │
                                                      ┌────────┴────────┐
                                                      ▼                 ▼
                                              Phase 4: Logit     Phase 5: Feature
                                              (선택적)            (실험적)
```

- **Path A**: AIHub 원본 데이터를 직접 샘플링하여 SFT. Teacher GPU 불필요.
- **Path B**: Teacher 모델로 합성 데이터를 생성한 뒤 필터링 후 SFT. 더 높은 품질.

---

## 사전 준비

```bash
pip install -r requirements.txt
```

| 항목 | 확인 방법 | 비고 |
|------|----------|------|
| GPU (CUDA) | `nvidia-smi` | SFT: 4xA100 80G 권장 |
| Hugging Face 접근 | `huggingface-cli whoami` | 필요 시 `HF_TOKEN` 설정 |
| torch CUDA | `python -c "import torch; print(torch.cuda.is_available())"` | |
| vLLM (Path B) | `python -c "import vllm"` | FP8 Teacher 사용 시 필요 |
| PaddleOCR | `python -c "from paddleocr import PaddleOCR"` | Path A 샘플링에 필요 |

**자동 설정 환경변수** (`run_distill.sh`가 자동으로 설정하므로 직접 지정 불필요):

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `TOKENIZERS_PARALLELISM` | `false` | tokenizer 병렬화 경고 방지 |
| `WANDB_MODE` | `disabled` | W&B 로깅 비활성화 |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | vLLM CUDA worker 호환 (`fork` → `spawn` 강제) |
| `CUDA_VISIBLE_DEVICES` | `0,1,...,N-1` | `NUM_GPUS` 기준 자동 설정 |

---

## 실행 경로 선택

| | Path A: Direct SFT | Path B: Full Distill | Path C: 독립 학습 |
|---|---|---|---|
| **설명** | AIHub 샘플 → 바로 SFT | Teacher 합성 → Filter → SFT | 기존 데이터로 QLoRA |
| **Teacher 필요** | 아니오 | 예 (235B, 4xA100) | 아니오 |
| **GPU 요구** | 4xA100 (SFT만) | 4xA100 이상 | 4xA100 |
| **용도** | 빠른 검증, 베이스라인 | 최고 품질 증류 | 커스텀 데이터 학습 |
| **대표 Config** | `distill_config_test.yaml` | `distill_config_test_output_lora_3000.yaml` | `training_config.yaml` |

---

## Phase 0: Prepare (AIHub 압축 해제)

AIHub 원천/라벨링 ZIP 파일을 `data/extracted/aihub/` 하위로 해제한다.

**입력** `../aihub_table_data/Training/` (ZIP 파일들)
**출력** `./data/extracted/aihub/Training/01.원천데이터/`, `02.라벨링데이터/`

```bash
PHASE=prepare \
CONFIG=config/distill_config_test_output_lora_3000.yaml \
EXTRACT_MAX_ZIPS=1 \
NUM_GPUS=4 \
bash distill/run_distill.sh
```

| 설정 | 설명 |
|------|------|
| `EXTRACT_MAX_ZIPS` | 해제할 ZIP 최대 개수 (테스트 시 `1`) |
| `AIHUB_TRAIN_DIR` | 원본 ZIP 경로 (기본: `../aihub_table_data/Training`) |
| `AIHUB_EXTRACT_DIR` | 해제 대상 경로 (기본: `./data/extracted/aihub`) |

---

## Phase 1: Generate (Teacher 합성 데이터 생성)

Teacher 모델로 테이블 이미지를 추론하여 HTML 구조 인식 결과를 합성한다.

**입력** `./data/extracted/aihub/Training/01.원천데이터` (이미지)
**출력** Config의 `synthetic_generation.output_path` (예: `./data/distill/modes/output_lora_3000/synthetic_raw.jsonl`)

```bash
PHASE=generate \
CONFIG=config/distill_config_test_output_lora_3000.yaml \
NUM_GPUS=4 \
bash distill/run_distill.sh
```

| 설정 키 | 기본값 | 설명 |
|---------|--------|------|
| `teacher.name_or_path` | `Qwen3-VL-235B-A22B-Instruct-FP8` | Teacher 모델 경로 |
| `teacher.backend` | `vllm` | `vllm` (FP8 권장) 또는 `transformers` |
| `teacher.tensor_parallel_size` | `4` | vLLM GPU 분할 수 |
| `synthetic_generation.target_samples` | `3000` | 목표 합성 데이터 수 (달성 시 종료) |
| `synthetic_generation.num_generations_per_image` | `1` | 이미지당 생성 횟수 (>1이면 consistency check) |
| `synthetic_generation.temperature` | `0.0` | 생성 다양성 (0.0 = greedy) |
| `synthetic_generation.enable_thinking` | `true` | thinking chain 포함 여부 |

주요 기능:
- **Consistency check**: `num_generations_per_image > 1`이면 동일 이미지에 대해 여러 번 생성 후 TEDS 점수로 최적 응답 선택
- **OCR 통합**: `prompting.style: chandra_table_with_ocr` 사용 시 OCR 정보를 프롬프트에 포함
- **진행률 추적**: `target_samples` 달성 시 자동 종료, `MAX_IMAGES` 환경변수로 상한 제한 가능

---

## Phase 2: Filter (품질 필터링 & 밸런싱)

Teacher 합성 데이터에서 저품질 샘플을 제거하고, span 복잡도 분포를 밸런싱한다.

**입력** `synthetic_raw.jsonl`
**출력** `synthetic_filtered.jsonl` → `train.jsonl` + `eval.jsonl` (학습 포맷 자동 변환)

```bash
PHASE=filter \
CONFIG=config/distill_config_test_output_lora_3000.yaml \
NUM_GPUS=4 \
bash distill/run_distill.sh
```

**6단계 필터링 파이프라인:**

| 단계 | 필터 | 기본 기준 |
|------|------|----------|
| 1 | HTML 유효성 | 파싱 가능, `require_valid_html: true` |
| 2 | 테이블 크기 | 2~80행, 2~30열 |
| 3 | Thinking chain | `require_thinking: false` (선택적) |
| 4 | Consistency score | `min_consistency_teds: 0.0` (테스트), `0.8` (프로덕션) |
| 5 | HTML 길이 | `max_html_length: 20000` |
| 6 | 중복 제거 | 해시 기반 |

**복잡도 밸런싱** (`target_distribution`):

| 복잡도 | 기준 | 기본 비율 (테스트) | 프로덕션 비율 |
|--------|------|-------------------|-------------|
| simple | span 없음 | 40% | 20% |
| medium | span 1~2개 | 35% | 30% |
| complex | span 3+개 | 25% | 50% |

---

## Phase 3: SFT (Student QLoRA 학습)

필터링된 데이터로 Student 모델을 Supervised Fine-tuning한다.

**입력** `train.jsonl`, `eval.jsonl`
**출력** `output/<mode>/student_sft/final/` (LoRA adapter 또는 병합 모델)

```bash
PHASE=sft \
CONFIG=config/distill_config_test_output_lora_3000.yaml \
NUM_GPUS=4 \
bash distill/run_distill.sh
```

| 설정 키 | 기본값 | 설명 |
|---------|--------|------|
| `student.name_or_path` | `Qwen3-VL-8B-Instruct` | Student 모델 |
| `student_sft.use_lora` | `true` | QLoRA 사용 여부 |
| `student_sft.lora_r` | `64` | LoRA rank (프로덕션: `128`) |
| `student_sft.lora_alpha` | `128` | LoRA alpha (프로덕션: `256`) |
| `student_sft.learning_rate` | `2.0e-5` | 학습률 |
| `student_sft.num_train_epochs` | `1` | 학습 에포크 (프로덕션: `3`) |
| `student_sft.max_seq_length` | `4096` | 최대 시퀀스 길이 (프로덕션: `8192`) |
| `student_sft.per_device_train_batch_size` | `1` | GPU당 배치 크기 |
| `student_sft.gradient_accumulation_steps` | `8` | gradient 누적 |

Multi-GPU 시 `torchrun --nproc_per_node=N`으로 DDP 학습이 자동 적용된다.

---

## Phase 4: Logit Distillation (선택적)

Teacher의 output logit 분포를 Student가 학습한다. SFT 결과 위에 추가 학습.

**입력** SFT 체크포인트 (`student_init_path`) + 학습 데이터
**출력** `output/<mode>/student_logit_distill/`

```bash
PHASE=logit \
CONFIG=config/distill_config_test_logit_lora.yaml \
NUM_GPUS=4 \
bash distill/run_distill.sh
```

| 설정 키 | 기본값 | 설명 |
|---------|--------|------|
| `logit_distillation.enabled` | `false` | `true`로 변경 시 활성화 |
| `logit_distillation.online_mode` | `false` | `false`: offline (사전 계산), `true`: online (실시간) |
| `logit_distillation.temperature` | `2.0` | soft target temperature |
| `logit_distillation.alpha` | `0.5` | distill loss 가중치 (1-alpha = hard loss) |
| `logit_distillation.save_top_k_logits` | `50` | offline 모드에서 top-k만 저장 (디스크 절약) |

활성화하려면 Config에서 `logit_distillation.enabled: true`로 설정해야 한다.
`distill_config_test_logit_lora.yaml`이 이 설정이 활성화된 예시 Config이다.

---

## Phase 5: Feature Distillation (실험적)

Teacher와 Student의 중간 레이어 feature를 MSE loss로 정렬한다.

**입력** SFT 체크포인트 + 학습 데이터
**출력** `output/<mode>/student_feature_distill/`

```bash
PHASE=feature \
CONFIG=config/distill_config_test_feature_lora.yaml \
NUM_GPUS=4 \
bash distill/run_distill.sh
```

| 설정 키 | 기본값 | 설명 |
|---------|--------|------|
| `feature_distillation.enabled` | `false` | `true`로 변경 시 활성화 |
| `feature_distillation.layer_mapping` | `{12:4, 24:8, 47:16, 70:24, 94:32}` | Teacher→Student 레이어 매핑 (235B: 94층, 8B: 32층) |
| `feature_distillation.projection_type` | `linear` | 차원 변환 방식 (`linear` / `mlp`) |
| `feature_distillation.teacher_hidden_dim` | `8192` | Teacher hidden 차원 |
| `feature_distillation.student_hidden_dim` | `4096` | Student hidden 차원 |

Loss 가중치: `feature: 0.3` + `logit: 0.5` + `hard: 0.2`

Teacher + Student를 동시에 로드하므로 single-process로만 실행된다.
`distill_config_test_feature_lora.yaml`이 이 설정이 활성화된 예시 Config이다.

---

## Evaluation

학습된 모델의 성능을 평가한다.

```bash
# run_distill.sh 통합 실행
PHASE=eval CONFIG=config/distill_config_test_output_lora_3000.yaml NUM_GPUS=4 bash distill/run_distill.sh

# 또는 독립 실행
bash eval/run_eval.sh --model output/output_lora_3000/student_sft/final --test_data data/processed/eval.jsonl

# vLLM 고속 평가 (전체 GPU 사용)
BACKEND=vllm TP_SIZE=4 BATCH_SIZE=16 MAX_NEW_TOKENS=1024 \
bash eval/run_eval.sh --model output/output_lora_3000/student_sft/final --test_data data/processed/eval.jsonl
```

| 메트릭 | 설명 |
|--------|------|
| **TEDS** | Tree-Edit-Distance-based Similarity. 구조 + 내용 유사도 |
| **TEDS-Structure** | 셀 내용 제외, 순수 구조만 비교 |
| **Span Cell F1** | colspan/rowspan 셀의 정밀도/재현율 |
| **Span Attribute Accuracy** | span 값(숫자)의 정확도 |
| **Simple Table Accuracy** | 단순 테이블 성능 유지 여부 |

평가 결과: `eval_results/predictions.jsonl`, `eval_results/metrics.json`, `eval_results/report.html`

고속 평가 권장값:
- `BACKEND=vllm`
- `TP_SIZE=<가용 GPU 수>` (예: 4)
- `BATCH_SIZE=8~32` (OOM 나면 감소)
- `MAX_NEW_TOKENS=512~1024`

### 모델 3종 비교 결과 (2026-02-13)

동일 평가셋(`data/distill/modes/output_lora_3000/eval.jsonl`, 126 samples)으로 아래 3개 모델을 비교했다.

- Teacher: `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8`
- Student (원본): `Qwen/Qwen3-VL-8B-Instruct`
- Student (학습): `output/output_lora_3000/student_sft/final`

평가 출력 경로:
- `eval_results/model_compare_20260213_134750/teacher/metrics.json`
- `eval_results/model_compare_20260213_134750/student_base/metrics.json`
- `eval_results/model_compare_20260213_134750/student_trained/metrics.json`

| 모델 | backend | batch | avg_teds | avg_teds_structure | avg_span_f1 | avg_position_f1 | avg_attribute_accuracy | complex_teds | avg_inference_time(s) |
|------|---------|-------|----------|--------------------|-------------|------------------|------------------------|--------------|-----------------------|
| Teacher (235B FP8) | vllm | 4 | 0.7388 | 0.7753 | 0.7430 | 0.8075 | 0.7976 | 0.7028 | 4.1938 |
| Student 원본 (8B) | vllm | 16 | 0.5960 | 0.6933 | 0.5614 | 0.6424 | 0.6257 | 0.6468 | 0.4334 |
| Student 학습 (8B LoRA) | vllm | 12 | 0.6016 | 0.6864 | 0.4858 | 0.5831 | 0.5365 | 0.6101 | 0.7705 |

해석:
- Teacher가 전체 품질 지표에서 가장 높다.
- 학습 Student는 원본 Student 대비 `avg_teds`가 소폭 상승(+0.0057)했다.
- 반면 `avg_span_f1`, `avg_position_f1`, `avg_attribute_accuracy`, `complex_teds`는 원본 Student 대비 하락했다.
- 추론 속도는 원본 Student가 가장 빠르다.

재현 명령:

```bash
# 1) Teacher
BACKEND=vllm TP_SIZE=4 BATCH_SIZE=4 MAX_NEW_TOKENS=1024 \
bash eval/run_eval.sh \
  --model Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --test_data data/distill/modes/output_lora_3000/eval.jsonl \
  --output_dir eval_results/model_compare_20260213_134750/teacher

# 2) Student (원본)
BACKEND=vllm TP_SIZE=4 BATCH_SIZE=16 MAX_NEW_TOKENS=1024 \
bash eval/run_eval.sh \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --test_data data/distill/modes/output_lora_3000/eval.jsonl \
  --output_dir eval_results/model_compare_20260213_134750/student_base

# 3) Student (학습)
BACKEND=vllm TP_SIZE=4 BATCH_SIZE=12 MAX_NEW_TOKENS=1024 \
bash eval/run_eval.sh \
  --model output/output_lora_3000/student_sft/final \
  --test_data data/distill/modes/output_lora_3000/eval.jsonl \
  --output_dir eval_results/model_compare_20260213_134750/student_trained
```

참고:
- `Qwen/Qwen3-VL-235B-A22B-Instruct`(비-FP8)는 현재 서버 4xH100 80GB + vLLM 설정에서 로딩 OOM이 발생했다.
- 235B 원본 모델 평가는 FP8 체크포인트(`...Instruct-FP8`) 기준으로 수행했다.

---

## Config 파일 가이드

### Config 비교

| Config 파일 | 용도 | Teacher | 샘플 수 | Backend | LoRA |
|------------|------|---------|---------|---------|------|
| `distill_config.yaml` | 프로덕션 | 235B | 30,000 | transformers | r=128 |
| `distill_config_test.yaml` | 스모크 테스트 | 235B-FP8 | 64 | vllm | r=64 |
| `distill_config_test_output_lora.yaml` | Output+LoRA 테스트 | 235B | 64 | transformers | r=64 |
| `distill_config_test_output_lora_3000.yaml` | 중규모 테스트 | 235B-FP8 | 3,000 | vllm | r=64 |
| `distill_config_test_output_fullft.yaml` | Full FT 테스트 | 235B | 64 | transformers | 미사용 |
| `distill_config_test_logit_lora.yaml` | Logit 증류 테스트 | 235B | 64 | transformers | r=64 |
| `distill_config_test_feature_lora.yaml` | Feature 증류 테스트 | 235B | 64 | transformers | r=64 |

### 선택 가이드

1. **첫 실행 (파이프라인 동작 확인)**: `distill_config_test_output_lora.yaml` (64개, 빠른 검증)
2. **중규모 실험**: `distill_config_test_output_lora_3000.yaml` (3000개, vLLM+FP8)
3. **프로덕션**: `distill_config.yaml` (30000개, 높은 consistency 기준)

### 핵심 설정 키

| 키 | 설명 |
|----|------|
| `teacher.name_or_path` | Teacher 모델 ID (HF hub 또는 로컬 경로) |
| `student.name_or_path` | Student 모델 ID |
| `synthetic_generation.target_samples` | 합성 데이터 목표 수량 |
| `student_sft.train_file` / `eval_file` | 학습/평가 데이터 경로 |
| `prompting.style` | 프롬프트 스타일 (`chandra_table_with_ocr` 등) |

---

## 주요 산출물

### Path A (Direct SFT)

```text
data/processed/
├── aihub_training_sampled_ocr_3000.jsonl     # 샘플링 결과
├── aihub_training_sampled_ocr_3000.report.json
├── train.jsonl                                # 학습 데이터
└── eval.jsonl                                 # 평가 데이터

output/student_sft_test/final/                 # SFT 모델
```

### Path B (Full Distill)

```text
data/distill/modes/output_lora_3000/
├── synthetic_raw.jsonl                        # Teacher 합성 원본
├── synthetic_filtered.jsonl                   # 필터링 결과
├── train.jsonl                                # 학습 데이터
└── eval.jsonl                                 # 평가 데이터

output/output_lora_3000/student_sft/final/     # SFT 모델
```

산출물 경로 패턴: Config 이름에서 `distill_config_test_` 접두사를 제거한 부분이 `modes/<mode>/`에 대응한다.
예: `distill_config_test_output_lora_3000.yaml` → `data/distill/modes/output_lora_3000/`

---

## 트러블슈팅

| # | 증상 | 원인 | 해결 |
|---|------|------|------|
| 1 | `No images found` | 이미지 경로 불일치 | `data/extracted/aihub/Training/01.원천데이터` 경로 확인. `PHASE=prepare` 재실행 |
| 2 | `Train dataset not found` | `student_sft.train_file` 경로 오류 | Config의 `train_file`/`eval_file`이 실제 파일 위치와 일치하는지 확인 |
| 3 | vLLM `Cannot re-initialize CUDA in forked subprocess` | fork 방식 비호환 | `run_distill.sh`가 자동으로 `VLLM_WORKER_MULTIPROC_METHOD=spawn` 설정. 수동 실행 시 직접 export |
| 4 | `_ARRAY_API not found` / `numpy.dtype size changed` | NumPy ABI 충돌 (vLLM은 NumPy 2.x, 기본 환경은 1.x) | `pip install -U --force-reinstall "numpy==2.2.6" "opencv-python-headless==4.13.0.92"` |
| 5 | `libGL.so.1: cannot open` | 컨테이너에 시스템 라이브러리 누락 | `apt-get install -y libgl1-mesa-glx libglib2.0-0` 또는 `opencv-python-headless` 사용 |
| 6 | CUDA OOM | 모델/배치가 GPU 메모리 초과 | `max_new_tokens`, `per_device_train_batch_size` 축소. 테스트 Config 먼저 사용 |
| 7 | `PHASE` 미지정 시 전체 재실행 | `PHASE` 기본값이 `all` | 특정 단계만 실행하려면 반드시 `PHASE=sft` 등으로 지정 |
| 8 | `No student checkpoint found` (eval) | SFT 미완료 또는 경로 불일치 | `output/` 하위에 체크포인트가 있는지 확인 |
| 9 | vLLM `tensor_parallel_size` 오류 | GPU 수 부족 | `teacher.tensor_parallel_size`를 가용 GPU 수 이하로 설정 |
| 10 | `PaddleOCR` import 오류 | PaddlePaddle 미설치 | `pip install paddleocr paddlepaddle` (GPU 환경: `paddlepaddle-gpu`) |

---

## 빠른 참조

### Path A: Direct SFT (Teacher 불필요)

```bash
# 1. 압축 해제
PHASE=prepare \
CONFIG=config/distill_config_test_output_lora_3000.yaml \
EXTRACT_MAX_ZIPS=1 NUM_GPUS=4 \
bash distill/run_distill.sh

# 2. 3000개 샘플링 + OCR
python -m data.sample_aihub_with_ocr \
  --input_dir ./data/extracted/aihub/Training \
  --output ./data/processed/aihub_training_sampled_ocr_3000.jsonl \
  --sample_count 3000 \
  --min_span_ratio 0.70 \
  --prompt_style chandra_table_with_ocr \
  --bbox_scale 1024

# 3. Train/Eval 분리
python -m data.split_dataset \
  --input ./data/processed/aihub_training_sampled_ocr_3000.jsonl \
  --output_dir ./data/processed \
  --eval_ratio 0.1

# 4. SFT 실행
PHASE=sft CONFIG=config/distill_config_test.yaml NUM_GPUS=4 bash distill/run_distill.sh
```

### Path B: Full Distill (Teacher 합성 → Filter → SFT)

```bash
# 1. 압축 해제
PHASE=prepare \
CONFIG=config/distill_config_test_output_lora_3000.yaml \
EXTRACT_MAX_ZIPS=1 NUM_GPUS=4 \
bash distill/run_distill.sh

# 2. 전체 파이프라인 (generate → filter → sft → eval)
CONFIG=config/distill_config_test_output_lora_3000.yaml \
NUM_GPUS=4 \
bash distill/run_distill.sh

# 또는 단계별 실행
PHASE=generate CONFIG=config/distill_config_test_output_lora_3000.yaml NUM_GPUS=4 bash distill/run_distill.sh
PHASE=filter   CONFIG=config/distill_config_test_output_lora_3000.yaml NUM_GPUS=4 bash distill/run_distill.sh
PHASE=sft      CONFIG=config/distill_config_test_output_lora_3000.yaml NUM_GPUS=4 bash distill/run_distill.sh
PHASE=eval     CONFIG=config/distill_config_test_output_lora_3000.yaml NUM_GPUS=4 bash distill/run_distill.sh
```

### Path C: 독립 학습 (train/run_train.sh)

```bash
# 데이터 검증 + QLoRA 학습 + 평가
NUM_GPUS=4 bash train/run_train.sh

# Phase 선택 (phase1 / phase2 / both)
PHASE=phase1 NUM_GPUS=4 bash train/run_train.sh

# LoRA 병합
MERGE=true NUM_GPUS=4 bash train/run_train.sh
```

---

## 비교 실험 파이프라인 (run_experiments.sh)

3가지 학습 방법 × 2가지 OCR 설정 = **6가지 실험**을 체계적으로 비교하는 파이프라인.

### 실험 매트릭스

| ID | 실험명 | 학습 방법 | OCR | Config 파일 | Teacher 필요 |
|---|---|---|---|---|---|
| E1 | SFT + OCR on | SFT only | ON | `exp_sft_ocr_on.yaml` | No |
| E2 | SFT + OCR off | SFT only | OFF | `exp_sft_ocr_off.yaml` | No |
| E3 | Response Distill + OCR on | generate→filter→sft | ON | `exp_resp_ocr_on.yaml` | Yes |
| E4 | Response Distill + OCR off | generate→filter→sft | OFF | `exp_resp_ocr_off.yaml` | Yes |
| E5 | Logit KD + OCR on | E3 + logit distill | ON | `exp_logit_ocr_on.yaml` | Yes |
| E6 | Logit KD + OCR off | E4 + logit distill | OFF | `exp_logit_ocr_off.yaml` | Yes |

베이스라인: B1 (Teacher 235B zero-shot), B2 (원본 Student 8B zero-shot)

### 실행 방법

마스터 스크립트 `run_experiments.sh`를 `PHASE=` 환경변수로 제어한다.

```bash
cd vlm_train/qwen3_vl_tsr

# 전체 실행 (Step 0~9 순차)
PHASE=all NUM_GPUS=4 bash run_experiments.sh

# 단계별 개별 실행
PHASE=prepare     bash run_experiments.sh   # Step 0: AIHub 압축 해제
PHASE=shared_data bash run_experiments.sh   # Step 1: 공유 데이터 3000개 생성
PHASE=e1          bash run_experiments.sh   # Step 2: E1 (SFT + OCR on)
PHASE=e2          bash run_experiments.sh   # Step 3: E2 (SFT + OCR off)
PHASE=e3          bash run_experiments.sh   # Step 4: E3 (Response Distill + OCR on)
PHASE=e5          bash run_experiments.sh   # Step 5: E5 (Logit KD, E3 기반)
PHASE=e4          bash run_experiments.sh   # Step 6: E4 (Response Distill + OCR off)
PHASE=e6          bash run_experiments.sh   # Step 7: E6 (Logit KD, E4 기반)
PHASE=baseline    bash run_experiments.sh   # Step 8: 베이스라인 평가
PHASE=eval_all    bash run_experiments.sh   # 전체 실험 평가 (E1~E6)
PHASE=aggregate   bash run_experiments.sh   # Step 9: 결과 집계 표 생성
```

### 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `PHASE` | `all` | 실행할 단계 |
| `NUM_GPUS` | `4` | GPU 수 |
| `SEED` | `42` | 랜덤 시드 (데이터 분할, 샘플링 고정) |
| `SAMPLE_COUNT` | `3000` | 샘플링할 데이터 수 |
| `MASTER_PORT` | `29510` | torchrun 마스터 포트 |
| `AIHUB_TRAIN_DIR` | `../aihub_table_data/Training` | AIHub ZIP 원본 경로 |
| `EXTRACT_MAX_ZIPS` | (전체) | 해제할 ZIP 최대 개수 |

### 실행 순서 및 의존성

```text
Step 0: prepare ─────────────────────────────────────────────────
Step 1: shared_data ─────────────────────────────────────────────
                │
     ┌──────────┼──────────┐
     ▼          ▼          ▼
  Step 2: E1  Step 3: E2  Step 4: E3 ──► Step 5: E5
  (SFT+OCR)   (SFT-OCR)  (Resp+OCR)     (Logit+OCR)
                           │
                           Step 6: E4 ──► Step 7: E6
                           (Resp-OCR)     (Logit-OCR)
                                          │
Step 8: baseline ◄────────────────────────┘
Step 9: eval_all + aggregate
```

- E1/E2는 Teacher 불필요, 즉시 실행 가능
- E5는 반드시 E3 완료 후 실행 (데이터 + SFT 체크포인트 복사)
- E6은 반드시 E4 완료 후 실행

### 데이터 구조

공정 비교를 위해 **동일 이미지 3000장, 동일 train/eval split** 사용.

```text
data/experiments/
  shared/                          # E1, E2가 직접 사용
    sampled_3000.jsonl             # 원본 샘플링 결과 (OCR 포함)
    train.jsonl, eval.jsonl        # OCR on (split)
    train_ocr_off.jsonl            # OCR off (프롬프트만 변환)
    eval_ocr_off.jsonl
  e3_resp_ocr_on/                  # E3 Teacher 합성 데이터
    synthetic_raw.jsonl → synthetic_filtered.jsonl → train.jsonl, eval.jsonl
  e4_resp_ocr_off/                 # E4 Teacher 합성 데이터
  e5_logit_ocr_on/                 # E5 (E3 복사 + teacher_logits/)
  e6_logit_ocr_off/                # E6 (E4 복사 + teacher_logits/)

output/
  e1_sft_ocr_on/student_sft/final/
  e2_sft_ocr_off/student_sft/final/
  e3_resp_ocr_on/student_sft/final/
  e4_resp_ocr_off/student_sft/final/
  e5_logit_ocr_on/student_logit_distill/final/
  e6_logit_ocr_off/student_logit_distill/final/

eval_results/
  e1_sft_ocr_on/metrics.json
  ...
  baseline_student_ocr_on/metrics.json
  baseline_teacher_ocr_on/metrics.json
  comparison_table.json            # 전체 비교 표
```

### 유틸리티 스크립트

| 스크립트 | 용도 | 사용 예 |
|---------|------|---------|
| `data/convert_prompt_style.py` | OCR on↔off 프롬프트 변환 | `python -m data.convert_prompt_style --input train.jsonl --output train_ocr_off.jsonl --prompt_style chandra_table_without_ocr` |
| `eval/aggregate_results.py` | 실험 결과 집계 비교 표 | `python -m eval.aggregate_results --results_dir eval_results/` |

### 결과 비교 표

`PHASE=aggregate`를 실행하면 다음과 같은 비교 표가 출력된다:

```
| Model                    | Method        | OCR  | TEDS   | TEDS-S | Span F1 | Attr Acc | Simple | Medium | Complex |
|--------------------------|---------------|------|--------|--------|---------|----------|--------|--------|---------|
| B2: Student 원본         | zero-shot     | on   | 0.xxxx | 0.xxxx | ...     | ...      | ...    | ...    | ...     |
| E1: SFT                  | SFT           | on   | 0.xxxx | 0.xxxx | ...     | ...      | ...    | ...    | ...     |
| E5: Logit KD             | Logit KD      | on   | 0.xxxx | 0.xxxx | ...     | ...      | ...    | ...    | ...     |
| B1: Teacher 235B         | zero-shot     | on   | 0.xxxx | 0.xxxx | ...     | ...      | ...    | ...    | ...     |
```

### 빠른 시작 예시

```bash
# 1. 압축 해제 (최초 1회)
PHASE=prepare EXTRACT_MAX_ZIPS=5 bash run_experiments.sh

# 2. 공유 데이터 생성
PHASE=shared_data bash run_experiments.sh

# 3. E1 (가장 빠른 실험, Teacher 불필요)
PHASE=e1 NUM_GPUS=4 bash run_experiments.sh

# 4. 결과 확인
PHASE=eval_all bash run_experiments.sh
PHASE=aggregate bash run_experiments.sh
```
