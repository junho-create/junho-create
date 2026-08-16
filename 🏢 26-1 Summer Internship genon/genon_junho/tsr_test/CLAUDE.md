# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TSR Labs는 Table Structure Recognition (TSR) 연구/개발 저장소로, TFLOP 모델(IJCAI 2024), 레이블링 도구, VLM 학습 파이프라인(Qwen3-VL), 합성 데이터 생성기, VLM 테이블 인식 테스트 도구를 포함한다.

## Repository Structure

- **TFLOP/** - TFLOP 모델 공식 구현체 (Swin Encoder + mBART Decoder + Layout Pointer)
- **tsr_lable_tool/** - PySide6 기반 테이블 데이터셋 레이블링 GUI 도구
- **train/vlm/** - Qwen3-VL 파인튜닝 및 지식 증류 파이프라인 (Teacher: 32B/235B, Student: 8B)
- **train/tflop/** - 커스텀 데이터셋 TFLOP 학습 스크립트, 체크포인트, 추론 유틸리티
- **train/aihub_table_data/** - AI Hub 테이블 인식 데이터셋 (Training/Validation 각 24개 zip)
- **test/test_model/** - VLM API 기반 테이블 인식 테스트 및 평가 도구 (PaddleOCR + LLM)
- **synthetic_tsr/** - 합성 테이블 데이터 생성기

## Common Commands

### TFLOP 모델

```bash
# 환경 설치
cd TFLOP && make install

# 코드 포매팅 (Black + isort)
cd TFLOP && make style_check

# 학습 (4 GPU, bf16, DeepSpeed ZeRO-2)
cd TFLOP && bash scripts/training/train_pubtabnet.sh

# 커스텀 데이터셋 학습 (pretrained 모델 파인튜닝)
cd train/tflop && bash scripts/training_tflop_add_51/train_51.sh

# 커스텀 데이터셋 평가
cd train/tflop && bash scripts/training_tflop_add_51/test_51.sh

# PubTabNet 평가
cd TFLOP && bash scripts/testing/test_pubtabnet.sh <bin_idx> <total_bin_cnt> <savedir> <epoch_step>
cd TFLOP && python evaluate_ted.py --model_inference_pathdir <savedir>/<epoch_step> --output_savepath <savedir>/<epoch_step>
```

### 레이블링 도구

```bash
cd tsr_lable_tool && uv sync && source .venv/bin/activate
cd label_tool && python tsr_label.py

# 데이터 정합성 검사
python util/check_jsonl.py --dataset_jsonl <path_to_dataset.jsonl>
```

### VLM 학습 (Qwen3-VL)

```bash
cd train/vlm && pip install -r requirements.txt

# Teacher QLoRA 학습
bash train/run_train.sh

# 증류 파이프라인 (전체 or 단계별)
bash distill/run_distill.sh
PHASE=generate bash distill/run_distill.sh  # Phase 1~5 개별 실행 가능

# 평가
bash eval/run_eval.sh

# 실험 일괄 실행
bash run_experiments.sh
```

### VLM 테이블 인식 테스트

```bash
cd test/test_model && uv sync && source .venv/bin/activate

# 테이블 이미지 테스트 (PaddleOCR + VLM API)
python test_table.py --input_dir "samples_png" --output_dir "output" --postprocess none

# 또는 셸 스크립트로 실행
bash test_table.sh
```

### 합성 데이터 생성

```bash
cd synthetic_tsr
python generate.py
python split_dataset.py
```

## Architecture

### TFLOP 모델 구조

`Image → Swin Vision Encoder → (B, 576, 1024) → mBART Decoder + Pointer Head → HTML/OTSL 시퀀스 + Cell-Text 매핑`

핵심 컴포넌트:
- **Visual Encoder**: `tflop/model/visual_encoder/swin.py` - Swin Transformer, 768x768 입력 → 24x24 패치
- **Decoder**: `tflop/model/decoder/mbart_decoder.py` - mBART 기반, Pointer Mechanism으로 셀-텍스트 정렬
- **Model**: `tflop/model/model/TFLOP.py` - 인코더+디코더 통합, ROI Alignment, HiMulConET
- **Loss**: `tflop/loss.py` - HTML/OTSL loss + Row/Col-wise Contrastive Learning
- **Lightning Module**: `tflop/lightning_module/lightning_module.py` - 학습/검증 루프

설정은 OmegaConf YAML (`config/exp_configs/`)로 관리되며 CLI에서 `key=value` 형식으로 오버라이드 가능.

### VLM 파이프라인 (Qwen3-VL)

Teacher-Student 지식 증류 기반 테이블 구조 인식 파이프라인:
- **Teacher**: Qwen3-VL-32B-Instruct (또는 235B-A22B-Instruct)
- **Student**: Qwen3-VL-8B-Instruct
- **학습 방식**: QLoRA (rank=128, alpha=256), DeepSpeed ZeRO-3, Flash Attention 2
- **프롬프트 스타일**: `chandra_table_with_ocr` / `chandra_table_without_ocr`

핵심 모듈:
- **train/train_qlora.py** - Teacher QLoRA 학습 (2-Phase: 구조 기초 → Span 특화)
- **train/collator.py** - 데이터 collator
- **distill/** - 5단계 증류 파이프라인
- **eval/evaluate.py** - TEDS 기반 평가
- **data/** - AI Hub 데이터 변환, 증강, 검증 스크립트
- **utils/html_utils.py** - HTML 파싱/검증, **prompt_templates.py** - 프롬프트 생성

실험 설정은 `config/` 디렉토리의 YAML 파일로 관리:
- `training_config.yaml` - 기본 학습 설정
- `distill_config.yaml` - 증류 파이프라인 설정
- `exp_*.yaml` - 개별 실험 설정 (SFT/Logit/Response × OCR on/off, E6 Span 등)

### VLM 증류 파이프라인 (5단계)

1. **Generate**: Teacher 모델로 합성 데이터 생성 (`distill/teacher_generate.py`)
2. **Filter**: 품질 필터링 및 Span 분포 밸런싱 (`distill/quality_filter.py`)
3. **SFT**: Student 모델 Supervised Fine-tuning (`distill/student_sft.py`)
4. **Logit Distill**: Teacher logit 기반 증류 (`distill/logit_distill_trainer.py`)
5. **Feature Distill**: Feature-level 증류, 레이어 매핑 기반 (`distill/feature_distill_trainer.py`)

### VLM 테이블 인식 테스트 (test/test_model)

`테이블 이미지/PDF → PaddleOCR 텍스트 추출 → VLM API 호출 → HTML 테이블 생성 → 후보정(선택)`

핵심 모듈:
- **test_table.py** - 메인 테스트 스크립트 (VLM API 호출, 결과 HTML 생성)
- **paddleocr_extractor.py** - PaddleOCR 기반 텍스트/좌표 추출
- **preprocess_ext_text.py** - Docling 기반 문서 전처리
- **table_prompt.py** - OCR 텍스트 포함 프롬프트 생성
- **html_template.py** - 결과 HTML 렌더링 템플릿

VLM API 엔드포인트: `http://192.168.75.173:8010/v1/chat/completions` (자체 서빙)

### 데이터 형식 (JSONL)

각 행은 하나의 테이블 이미지에 대한 JSON 객체:
- `file_name`: 이미지 파일명
- `dr_coord`: OCR 감지 좌표 `{"0": [[[x1,y1,x2,y2]], cell_idx, "text"], ...}`
- `gold_coord`: GT 좌표 리스트 `["x1 y1 x2 y2 class_id text", ...]`
- `org_html`: 토큰화된 HTML 태그 배열 (`<thead>`, `<tr>`, `<td>`, ...)
- `otsl_seq`: 구조 토큰 배열 (`C-tag`, `L-tag`, `U-tag`, `X-tag`, `NL-tag`)
- `num_rows`, `num_cols`, `split`

**정합성 규칙**: gold_coord 셀 수 == HTML 셀 수 == OTSL 논리 셀 수 (불일치 시 학습 오류 발생)

### OTSL 토큰 의미

| 토큰 | 의미 |
|------|------|
| C-tag | 새 셀 |
| L-tag | colspan 연속 (좌측 병합) |
| U-tag | rowspan 연속 (상단 병합) |
| X-tag | colspan+rowspan 교차 병합 |
| NL-tag | 행 종료 |

## Code Style

- TFLOP: Black (line-length=88) + isort (profile=black), Python 3.10+
- 레이블링 도구: Python 3.11+, PySide6, uv 패키지 매니저
- VLM 학습: Python 3.11+, Qwen3-VL, DeepSpeed, Flash Attention 2
- 테스트 도구: Python 3.11+, PaddleOCR, PyMuPDF, uv 패키지 매니저
- 한국어 주석/문서 사용

## Key Hyperparameters

### TFLOP
- `max_length`: 시퀀스 최대 길이 (기본 1376, 커스텀 데이터셋에 따라 조정)
- `bbox_token_cnt`: bbox 토큰 수 (864)
- `input_size`: 이미지 해상도 (768x768)
- `lr`: 학습률 (PubTabNet: 8e-5, 파인튜닝: 1e-5)
- Contrastive Learning: `use_RowWise_contLearning`과 `use_ColWise_contLearning` 중 최소 하나 True 필수

### VLM (Qwen3-VL)
- `max_seq_length`: 8192
- `learning_rate`: Teacher 2e-5, Student 5e-5
- `image_resolution`: 2048 (고해상도 테이블)
- `lora_r` / `lora_alpha`: Student 128/256
- `temperature`: 증류 soft target 2.0, 생성 다양성 0.3
- 2-Phase 학습: Phase1 구조 기초(2 epoch, 2e-5) → Phase2 Span 특화(3 epoch, 5e-6, 2x upsampling)
