# AIHub 샘플링 파이프라인 사용 가이드

## 개요

AIHub 테이블 데이터를 분석 → 샘플링 → OCR 추가하는 3단계 파이프라인.
각 단계가 독립 모듈로 분리되어 있어 개별 실행 및 조합이 자유롭다.

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  analyze_aihub   │────▶│   sample_aihub   │────▶│     add_ocr      │
│                  │     │                  │     │                  │
│ AIHub 디렉토리   │     │ 인덱스 JSONL     │     │ *_raw.jsonl      │
│  → 인덱스 JSONL  │     │  → *_raw.jsonl   │     │  → *.jsonl (OCR) │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

## 파일 구조

```
train/vlm/
├── data/
│   ├── analyze_aihub.py     # Step 1: 데이터 분석 + 인덱싱
│   ├── sample_aihub.py      # Step 2: 복잡도 비율 샘플링 + 중복제거
│   ├── add_ocr.py           # Step 3: OCR 추가 + 프롬프트 변환
│   ├── run_sampling.sh      # 전체 파이프라인 쉘 스크립트
│   ├── convert_aihub.py     # (유틸) AIHub 원본 로드
│   └── convert_prompt_style.py  # (유틸) 프롬프트 스타일 변환
├── utils/
│   └── ocr_extractor.py     # PaddleOCRExtractor 독립 모듈
```

---

## Step 1: 데이터 분석 (`analyze_aihub.py`)

AIHub 전체 데이터를 순회하며 HTML을 파싱하고, 복잡도 분류 및 구조 서명을 포함한 인덱스 JSONL로 저장한다.

### 기본 사용법

```bash
cd train/vlm

python -m data.analyze_aihub \
  --input_dir ./data/extracted/aihub/Training \
  --output ./data/index/training_index.jsonl
```

### 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--input_dir` | `./data/extracted/aihub/Training` | 압축 해제된 AIHub 디렉토리 |
| `--output` | `./data/index/training_index.jsonl` | 출력 인덱스 JSONL 경로 |
| `--skip` | 활성 | output 파일이 이미 존재하면 재분석 생략 |
| `--no_skip` | - | `--skip` 무시하고 강제 재분석 |

### 출력 형식 (인덱스 JSONL)

각 행은 하나의 테이블에 대한 메타정보:

```json
{
  "file_id": "T01_C01_001234",
  "image_path": "/path/to/image.jpg",
  "html_path": "/path/to/table.html",
  "json_path": "/path/to/meta.json",
  "normalized_html": "<table><tr><td>...</td></tr></table>",
  "complexity": "complex",
  "complexity_score": 0.62,
  "num_rows": 5,
  "num_cols": 4,
  "num_span_cells": 3,
  "structure_signature": "a1b2c3d4e5f6...",
  "table_type": "병합표",
  "table_field": "행정",
  "has_header": true
}
```

### 구조 서명 (`structure_signature`)

테이블의 행·열 수, span 셀 수, 각 셀의 span 토큰 시퀀스를 결합한 SHA-1 해시.
구조적으로 동일한 테이블은 같은 서명을 갖게 되어, Step 2 샘플링 시 유사도 기반 중복제거에 사용된다.

### 참고

- 인덱스 파일은 한 번 생성하면 반복 사용 가능 (`--skip`이 기본값)
- 데이터 디렉토리 구조: `input_dir/01.원천데이터/`, `input_dir/02.라벨링데이터/`
- 파싱 실패 건은 건너뛰고 통계에 집계됨

---

## Step 2: 샘플링 (`sample_aihub.py`)

인덱스 JSONL을 기반으로 복잡도 비율에 따라 샘플링하여 학습용 JSONL을 생성한다.
두 종류의 중복제거가 적용된다:

1. **구조 유사도 중복제거** — 동일한 구조 서명을 가진 샘플을 제한하여 과적합 방지
2. **Split 간 중복제거** — `--exclude`로 train/val/test 간 이미지 겹침 방지

### 기본 사용법

```bash
python -m data.sample_aihub \
  --index ./data/index/training_index.jsonl \
  --output ./data/experiments/e7/train_raw.jsonl \
  --count 10000 \
  --ratio_complex 0.30 \
  --ratio_medium 0.40 \
  --ratio_simple 0.30 \
  --seed 42
```

### 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--index` | (필수) | 인덱스 JSONL 경로 |
| `--output` | (필수) | 출력 JSONL 경로 |
| `--count` | (필수) | 샘플 수 |
| `--ratio_complex` | `0.30` | complex 비율 |
| `--ratio_medium` | `0.40` | medium 비율 |
| `--ratio_simple` | `0.30` | simple 비율 |
| `--prompt_style` | `chandra_table_without_ocr` | 프롬프트 스타일 |
| `--hard_first` / `--no-hard_first` | 활성 | 어려운 샘플 우선 선택 |
| `--seed` | `42` | 랜덤 시드 |
| `--exclude` | - | 제외할 JSONL 경로들 (공백 구분) |
| `--max_per_signature` | `5` | 동일 구조 서명당 최대 샘플 수 (0=비활성화) |
| `--no_thinking` | - | thinking chain 생성 비활성화 |

### 중복제거

#### 구조 유사도 중복제거 (`--max_per_signature`)

동일한 `structure_signature`를 가진 샘플의 수를 제한한다.
구조가 유사한 샘플(같은 행·열 수, 같은 span 패턴)이 과도하게 포함되면 모델이 특정 패턴에 과적합될 수 있으므로, 기본값 5개로 제한한다.

```bash
# 기본: 동일 구조당 최대 5개
python -m data.sample_aihub --index idx.jsonl --output out.jsonl --count 10000

# 더 엄격하게 제한
python -m data.sample_aihub --index idx.jsonl --output out.jsonl --count 10000 \
  --max_per_signature 3

# 비활성화 (모든 샘플 허용)
python -m data.sample_aihub --index idx.jsonl --output out.jsonl --count 10000 \
  --max_per_signature 0
```

#### Split 간 중복제거 (`--exclude`)

train/val/test 간 이미지 중복을 방지한다. `--exclude`로 이미 샘플링된 JSONL을 지정하면, 해당 파일의 `metadata.image_path`를 추출하여 자동으로 필터링한다.

```bash
# test 먼저 → validation에서 test 제외 → train에서 test+val 제외
python -m data.sample_aihub --index idx.jsonl --output test_raw.jsonl --count 500 --seed 44
python -m data.sample_aihub --index idx.jsonl --output val_raw.jsonl  --count 1000 --seed 43 \
  --exclude test_raw.jsonl
python -m data.sample_aihub --index idx.jsonl --output train_raw.jsonl --count 10000 --seed 42 \
  --exclude test_raw.jsonl val_raw.jsonl
```

### 샘플링 처리 흐름

```
인덱스 로드 → file_id 중복 제거 → --exclude 적용 → 구조 유사도 필터
→ 복잡도별 풀 구성 → 비율 할당 → hard_first 정렬 → 선택 + backfill → 셔플 → 레코드 빌드
```

### 출력 형식

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": [
      {"type": "image", "image": "/path/to/image.jpg"},
      {"type": "text", "text": "프롬프트 (OCR-off 스타일)"}
    ]},
    {"role": "assistant", "content": "<think>...</think>\n<table>...</table>"}
  ],
  "metadata": {
    "image_path": "/path/to/image.jpg",
    "complexity": "complex",
    "prompt_style": "chandra_table_without_ocr"
  }
}
```

### 참고

- `--prompt_style`은 기본적으로 `chandra_table_without_ocr` (OCR 없는 상태로 생성)
- OCR이 필요한 경우 Step 3(`add_ocr.py`)에서 추가
- 비율의 합이 1.0이 아니어도 자동 정규화됨
- 리포트 JSON이 `<output>.report.json`에 자동 생성됨

---

## Step 3: OCR 추가 (`add_ocr.py`)

샘플링된 JSONL에 PaddleOCR로 OCR 정보를 추가하고, 프롬프트를 OCR-on 스타일로 변환한다.

### 기본 사용법

```bash
python -m data.add_ocr \
  --input ./data/experiments/e7/train_raw.jsonl \
  --output ./data/experiments/e7/train.jsonl \
  --prompt_style chandra_table_with_ocr \
  --bbox_scale 1024
```

### 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--input` | (필수) | 입력 JSONL 경로 |
| `--output` | (필수) | 출력 JSONL 경로 |
| `--prompt_style` | `chandra_table_with_ocr` | 변환 타겟 프롬프트 스타일 |
| `--bbox_scale` | `1024` | 바운딩박스 정규화 스케일 |
| `--ocr_lang` | `korean` | PaddleOCR 언어 코드 |

### 출력 형식

입력 레코드에 다음이 추가/변경된다:

```json
{
  "metadata": {
    "ocr_info": [{"text": "셀 텍스트", "bbox": [10, 20, 100, 50]}, ...],
    "bbox_scale": 1024,
    "prompt_style": "chandra_table_with_ocr"
  }
}
```

### 참고

- PaddleOCR 의존성 필요: `pip install paddleocr paddlepaddle`
- OCR 결과가 비어 있어도 레코드는 유지됨 (경고만 출력)
- 프롬프트 텍스트가 OCR 정보를 포함하는 스타일로 자동 변환됨

---

## 쉘 스크립트: `run_sampling.sh`

위 3단계를 자동으로 조합 실행하는 스크립트.

### 사용법

```bash
cd train/vlm

# 전체 파이프라인 (test → validation → train 순서)
bash data/run_sampling.sh all

# 개별 split만 실행
bash data/run_sampling.sh train
bash data/run_sampling.sh validation
bash data/run_sampling.sh test
```

### 설정 변경

스크립트 상단의 변수를 수정하여 설정을 조정한다:

```bash
# 데이터 경로
INPUT_DIR="./data/extracted/aihub/Training"
INDEX_FILE="./data/index/training_index.jsonl"
OUTPUT_DIR="./data/experiments/e7"

# 복잡도 비율
RATIO_COMPLEX=0.30
RATIO_MEDIUM=0.40
RATIO_SIMPLE=0.30

# 샘플 수
TRAIN_COUNT=10000
VAL_COUNT=1000
TEST_COUNT=500

# 프롬프트/OCR
PROMPT_STYLE="chandra_table_with_ocr"
BBOX_SCALE=1024
SEED=42
```

### 실행 흐름 (`all` 모드)

```
1. analyze_aihub  → training_index.jsonl (skip 가능)
2. sample_aihub   → test_raw.jsonl      (seed+2)
3. sample_aihub   → validation_raw.jsonl (seed+1, exclude: test)
4. sample_aihub   → train_raw.jsonl     (seed,   exclude: test+val)
5. add_ocr × 3    → train.jsonl, validation.jsonl, test.jsonl
```

`all` 모드에서는 test → validation → train 순서로 샘플링하며, 각 단계에서 `--exclude`로 이전 split의 결과를 제외하여 train/val/test 간 이미지 중복을 자동으로 방지한다.

---

## 유틸리티: `utils/ocr_extractor.py`

PaddleOCR를 래핑한 독립 모듈. `add_ocr.py` 외에도 다른 곳에서 직접 사용 가능.

```python
from utils.ocr_extractor import PaddleOCRExtractor, normalize_ocr_items

# 초기화 (lazy - 첫 extract 호출 시 모델 로드)
extractor = PaddleOCRExtractor(lang="korean", bbox_scale=1024)

# OCR 실행
items = extractor.extract("/path/to/image.jpg")
# → [{"text": "...", "bbox": [x0,y0,x1,y1], "score": 0.95}, ...]

# score 제거 (학습 데이터용)
clean = normalize_ocr_items(items, keep_score=False)
# → [{"text": "...", "bbox": [x0,y0,x1,y1]}, ...]
```

---

## 활용 예시

### E7 학습 데이터 구성

```bash
cd train/vlm

# 방법 1: 쉘 스크립트로 한 번에 실행
bash data/run_sampling.sh all

# 방법 2: 단계별 실행 (설정 커스터마이징)
python -m data.analyze_aihub \
  --input_dir ./data/extracted/aihub/Training \
  --output ./data/index/training_index.jsonl

python -m data.sample_aihub \
  --index ./data/index/training_index.jsonl \
  --output ./data/experiments/e7/train_raw.jsonl \
  --count 10000 \
  --ratio_complex 0.30 --ratio_medium 0.40 --ratio_simple 0.30

python -m data.add_ocr \
  --input ./data/experiments/e7/train_raw.jsonl \
  --output ./data/experiments/e7/train.jsonl
```

### OCR 없이 샘플링만 (OCR-off 학습용)

```bash
python -m data.analyze_aihub --input_dir ./data/extracted/aihub/Training --output idx.jsonl
python -m data.sample_aihub --index idx.jsonl --output train.jsonl --count 5000 \
  --prompt_style chandra_table_without_ocr
# add_ocr 단계 생략 → OCR-off 데이터 완성
```

### 기존 JSONL에 OCR만 추가

```bash
python -m data.add_ocr \
  --input ./existing_dataset.jsonl \
  --output ./existing_dataset_with_ocr.jsonl \
  --prompt_style chandra_table_with_ocr
```

### Medium 강화 샘플링 (E7 Phase 2용)

```bash
python -m data.sample_aihub \
  --index idx.jsonl --output phase2_train_raw.jsonl \
  --count 5000 \
  --ratio_complex 0.25 --ratio_medium 0.45 --ratio_simple 0.30 \
  --hard_first \
  --exclude phase1_train_raw.jsonl  # Phase 1 데이터 제외
```

---

## 중복제거 전략 요약

| 종류 | 목적 | 적용 위치 | 옵션 |
|------|------|-----------|------|
| file_id 중복 | 인덱스 내 동일 파일 제거 | `sample_aihub.py` 내부 | 자동 |
| 구조 유사도 | 동일 구조 패턴 과적합 방지 | `sample_aihub.py` | `--max_per_signature` (기본 5) |
| Split 간 중복 | train/val/test 이미지 겹침 방지 | `sample_aihub.py` | `--exclude` |

**구조 유사도 중복제거가 필요한 이유**: AIHub 데이터에는 동일한 행·열 수와 span 패턴을 가진 테이블이 다수 존재한다. 이들이 과도하게 포함되면 모델이 해당 패턴에만 최적화되어 다양한 테이블 구조에 대한 일반화 성능이 떨어진다. `--max_per_signature=5`로 동일 구조당 최대 5개만 허용하여 학습 데이터의 구조적 다양성을 확보한다.

---

## 이전 파이프라인과의 차이

| 항목 | 이전 (삭제됨) | 현재 |
|------|-------------|------|
| 분석+샘플링+OCR | `sample_aihub_with_ocr.py` 1파일 | 3개 독립 모듈 |
| OCR 없는 샘플링 | `sample_aihub_no_ocr.py` 별도 | `sample_aihub.py` + OCR 단계 생략 |
| Split 분할 | `build_sampling_splits.py` 별도 | `--exclude` 옵션으로 순차 샘플링 |
| OCR 추가 | `add_ocr_to_dataset.py` | `add_ocr.py` (동일 기능, 간소화) |
| 인덱스 | 없음 (매번 전수 분석) | `analyze_aihub.py` 1회 → 재사용 |
| 구조 유사도 중복제거 | `build_sampling_splits.py` 내 signature | `sample_aihub.py --max_per_signature` |
| Split 간 중복 방지 | history JSONL 기반 | `--exclude`로 image_path 기반 |
| 메타데이터 | 15+ 필드 | 3 필드 (image_path, complexity, prompt_style) |
