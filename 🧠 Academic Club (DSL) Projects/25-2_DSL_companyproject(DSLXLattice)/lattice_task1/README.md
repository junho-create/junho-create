# Contract Extractor (Text PDF → Ledger Fields)

PDF 형식의 계약서에서 구조화된 정보를 자동으로 추출하는 파이프라인 시스템입니다.

## 주요 기능

- 📄 PDF 계약서에서 텍스트 추출 (텍스트 PDF 및 OCR 지원)
- 🔍 계약명, 날짜, 금액 등 11개 필드 자동 추출
- 🤖 LLM 기반 날짜 추출 (체결일, 시작일, 종료일)
- 📊 각 필드에 대한 신뢰도 점수 및 근거(evidence) 제공
- ✅ 교차 검증 및 정규화 처리
- 📦 배치 처리 지원 (여러 PDF 파일 일괄 처리)
- 🧪 정답 파일 기반 성능 평가 도구

## 설치

```bash
cd /home/user/vscode/lattice_contract/project
python3 -m pip install -r requirements.txt
```

## 설정

OCR API와 LLM API를 사용하려면 설정이 필요합니다.

**빠른 설정:**
```bash
# 1. .env 파일 생성 (프로젝트 루트에)
touch .env

# 2. .env 파일 편집하여 실제 API URL 입력
# OCR_API_URL=https://your-actual-api-url.com/ocr
# BEDROCK_API_URL=https://your-actual-api-url.com/bedrock
# OCR_TIMEOUT=60
# BEDROCK_TIMEOUT=60
```

**설정 항목:**
- `OCR_API_URL`: OCR API 엔드포인트 URL (스캔 PDF 처리용)
- `BEDROCK_API_URL`: LLM API 엔드포인트 URL (날짜 추출용)
- `OCR_TIMEOUT`: OCR API 타임아웃 (초, 기본값: 60)
- `BEDROCK_TIMEOUT`: LLM API 타임아웃 (초, 기본값: 60)

**참고:**
- OCR API가 설정되지 않으면 텍스트 PDF만 처리 가능
- LLM API가 설정되지 않으면 날짜 추출이 Fallback 로직으로 동작

## 사용법

### CLI 사용

#### 단일 파일 처리
```bash
python3 cli.py samples/sample.pdf
```

결과는 `outputs/json/sample.json`에 저장됩니다.

#### 배치 처리 (여러 파일 일괄 처리)
```bash
# 방법 1: --batch 옵션 사용
python3 cli.py --batch

# 방법 2: --all 옵션 사용 (--batch와 동일)
python3 cli.py --all

# 방법 3: 폴더 경로 직접 지정
python3 cli.py samples/
```

배치 처리 시 `samples/` 폴더의 모든 PDF 파일을 처리하고, 각 파일의 결과를 `outputs/json/`에 저장합니다.

### Python 코드에서 사용
```python
from app.pipeline.run import run_on_file

result = run_on_file("samples/sample.pdf")
print(result)
```

### 성능 평가 (정답 파일과 비교)

정답 파일(`expected/`)과 추출 결과를 비교하여 성능을 평가할 수 있습니다.

```bash
# 단일 샘플 비교
python3 compare_results.py sample1

# 모든 샘플 비교
python3 compare_results.py --all

# 상세 정보 포함
python3 compare_results.py --all --detailed
```

## 추출 필드

시스템은 다음 11개 필드를 추출합니다:

- `TEMP_KEY`: 계약번호 (임시 키, 현재 미사용)
- `CNT_NAME`: 계약명
- `CNT_CON_DATE`: 계약 체결일 (YYYYMMDD 형식, LLM 기반 추출)
- `CNT_ST_DATE`: 계약 시작일 (YYYYMMDD 형식, LLM 기반 추출)
- `CNT_END_DATE`: 계약 종료일 (YYYYMMDD 형식, LLM 기반 추출)
- `CNT_CONCLUDED`: 체결 여부 (사용자 입력 필요, O/X)
- `CNT_RENEWAL`: 재계약 여부 (O/X)
- `CNT_AMT`: 계약 금액 (숫자만, 콤마 제거)
- `CNT_AMT_CRY`: 통화 (KRW/USD/EUR/JPY)
- `CNT_AUTO_RNW_TERM_AMT`: 자동 갱신 기간 수치
- `CNT_AUTO_RNW_TERM_UNIT`: 자동 갱신 기간 단위 (DAY/MONTH/YEAR)

## 출력 형식

각 필드는 다음과 같은 구조로 출력됩니다:

```json
{
  "CNT_CON_DATE": {
    "value": "20210324",
    "confidence": 1.0,
    "evidence": {
      "page": 1,
      "lines": [38],
      "snippet": "2021년 3월 24일"
    },
    "note": null
  }
}
```

- `value`: 추출된 값 (정규화된 형식)
- `confidence`: 신뢰도 점수 (0.0 ~ 1.0)
- `evidence`: 근거 정보 (페이지, 라인 번호, 스니펫)
- `note`: 검수 필요 여부 또는 기타 메모

## 프로젝트 구조

```
project/
├── app/                          # 메인 애플리케이션 코드
│   ├── extractors/              # 필드별 추출기 모듈
│   │   ├── base.py              # 추출기 베이스 클래스
│   │   ├── cnt_identifier.py   # 계약번호 추출 (현재 미사용)
│   │   ├── cnt_name.py         # 계약명 추출
│   │   ├── cnt_con_date.py     # 체결일 추출 (LLM 기반)
│   │   ├── cnt_period.py       # 기간 추출 (LLM 기반)
│   │   ├── cnt_amount.py       # 금액 추출
│   │   ├── cnt_auto_renewal.py # 자동갱신 추출
│   │   └── cnt_renewal_flag.py # 재계약 여부 추출
│   ├── ingest/                 # PDF 로딩 및 텍스트 처리
│   │   ├── loader_pdf.py       # PDF 텍스트 추출 및 OCR 폴백
│   │   ├── loader_ocr.py       # OCR API 연동 (PDF 분할 처리 포함)
│   │   ├── date_llm_extractor.py # LLM 기반 날짜 추출
│   │   └── detector_text.py   # 텍스트 감지 (확장용)
│   ├── models/                 # 데이터 모델
│   │   └── result.py           # 결과 모델
│   ├── pipeline/               # 파이프라인 오케스트레이션
│   │   ├── router.py           # 메인 라우터
│   │   ├── run.py              # 실행 래퍼
│   │   └── config.py           # 설정 (가중치, 임계치)
│   ├── postprocess/            # 후처리 (점수화, 정규화, 검증)
│   │   ├── scoring.py          # 점수화
│   │   ├── normalize.py        # 정규화
│   │   └── validate.py         # 검증 및 최종 선택
│   └── utils/                  # 공통 유틸리티
│       ├── regexes.py          # 정규식 패턴 모음
│       ├── text.py             # 텍스트 처리 유틸리티
│       └── profile.py          # 프로파일 관리 (현재 미사용)
├── cli.py                      # CLI 진입점
├── compare_results.py          # 성능 평가 스크립트
├── debug_*.py                  # 디버깅 스크립트들
├── expected/                   # 정답 파일 (ground truth)
├── outputs/json/               # 추출 결과 저장 디렉토리
├── samples/                    # 샘플 PDF 파일
├── requirements.txt           # Python 의존성 목록
└── README.md                   # 이 파일
```

## 처리 흐름

```
PDF 파일 입력
    ↓
[1단계: Ingest] PDF → TextBlock 리스트 변환
    ├─ OCR API 호출 시도 (우선)
    │   ├─ 5페이지 이하: 바로 OCR 처리
    │   └─ 5페이지 초과: 자동으로 5페이지 단위로 분할하여 OCR 처리
    │       └─ 각 분할 결과를 합치고 페이지 번호 보존
    └─ OCR 실패 시: pdfminer.six로 텍스트 추출 (폴백)
    ↓
[2단계: Extract] 각 필드별 추출기 실행 → Candidate 후보 수집
    ├─ 날짜 필드: LLM 기반 추출 (체결일, 시작일, 종료일)
    ├─ 기타 필드: 정규식 및 휴리스틱 기반 추출
    └─ LLM 실패 시 Fallback 로직 사용
    ↓
[3단계: Scoring] 후보들에 점수 부여 (출처, 키워드, 포맷 등)
    ↓
[4단계: Normalize] 날짜/금액/통화 형식 정규화
    ↓
[5단계: Validate] 최종 후보 선택 및 교차 검증
    ↓
JSON 결과 출력
```

## 주요 특징

### 1. 모듈화된 설계
- 각 필드 추출기가 독립적으로 동작
- 새로운 필드 추가가 용이한 구조

### 2. 방어적 프로그래밍
- 개별 모듈 오류가 전체 파이프라인을 중단시키지 않음
- 예외 처리 및 로깅 포함

### 3. 증거 기반 추출
- 모든 추출 값에 대해 페이지, 라인 번호, 스니펫 제공
- 검수 및 디버깅에 유용

### 4. 신뢰도 점수
- 각 필드에 대한 신뢰도 점수 제공
- 임계치 미만 시 "검수 필요" 플래그 자동 추가

### 5. 정규화 및 검증
- 날짜: YYYYMMDD 형식으로 통일 (다양한 입력 형식 지원)
- 금액: 숫자만 추출 (콤마 및 공백 제거)
- 통화: 대문자 3자리 코드로 변환 (KRW/USD/EUR/JPY)
- 교차 검증: 시작일 ≤ 종료일 등 논리 검증

### 6. LLM 기반 날짜 추출
- 체결일, 시작일, 종료일을 한 번의 LLM 호출로 추출
- 효력일/발효일 관련 키워드 인식
- LLM 실패 시 정규식 기반 Fallback 로직 사용
- 캐싱으로 동일 문서 재처리 시 비용 절감

### 7. OCR 지원
- **우선 OCR 시도**: 모든 PDF에 대해 먼저 OCR API를 호출하여 처리
- **자동 폴백**: OCR 실패 시 자동으로 텍스트 추출로 전환
- **대용량 PDF 분할 처리**: 네이버 클로바 OCR API 제한(최대 5페이지)에 맞춰 5페이지를 초과하는 PDF를 자동으로 분할하여 처리
- **페이지 번호 보존**: 분할된 PDF의 OCR 결과를 원본 PDF의 페이지 번호로 자동 매핑
- **임시 파일 관리**: 분할된 PDF 임시 파일을 자동으로 정리
- **OCR API 응답 변환**: OCR API 응답을 TextBlock 형식으로 변환하여 기존 파이프라인과 호환

## 정답 파일 및 평가

`expected/` 디렉토리에 각 샘플의 정답 파일을 저장하여 성능을 평가할 수 있습니다.

정답 파일 형식:
```json
{
  "TEMP_KEY": null,
  "CNT_NAME": "계약서명",
  "CNT_CON_DATE": "20210324",
  "CNT_ST_DATE": "20210101",
  "CNT_END_DATE": "20211231",
  ...
}
```

자세한 내용은 [expected/README.md](expected/README.md)를 참조하세요.

## 상세 문서

프로젝트의 전체 구조와 각 모듈의 역할에 대한 상세한 설명은 [PROJECT_GUIDE.md](PROJECT_GUIDE.md)를 참조하세요.

## 의존성

- `pdfminer.six`: PDF 텍스트 추출
- `python-dotenv`: 환경변수 관리
- `requests`: HTTP 요청 (OCR/LLM API 호출)
- `PyPDF2`: PDF 분할 처리 (대용량 PDF OCR 처리용)

## 디버깅 도구

프로젝트에는 디버깅을 위한 스크립트들이 포함되어 있습니다:

- `debug_text.py`: PDF에서 추출된 텍스트 블록 확인
- `debug_extractors.py`: 각 추출기의 후보 확인
- `debug_pipeline.py`: 전체 파이프라인 단계별 확인

사용 예시:
```bash
python3 debug_text.py samples/sample.pdf
python3 debug_extractors.py samples/sample.pdf
python3 debug_pipeline.py samples/sample.pdf
```

## 주의사항

- **OCR 우선 처리**: 모든 PDF에 대해 먼저 OCR API를 시도하며, 실패 시 텍스트 추출로 자동 폴백
- **대용량 PDF 분할**: 5페이지를 초과하는 PDF는 자동으로 분할하여 처리 (네이버 클로바 OCR API 제한)
- **OCR API 설정**: OCR API가 설정되지 않으면 텍스트 추출로 폴백하여 처리
- **LLM API**: 날짜 추출을 위해 LLM API가 필요하지만, 설정되지 않으면 Fallback 로직 사용
- **신뢰도 점수**: 0.75 미만이면 "검수 필요" 플래그 자동 추가
- **체결 여부**: `CNT_CONCLUDED`는 항상 사용자 입력 필요 (자동 추출 불가)

이 프로젝트는 내부 사용을 위한 것입니다.

## 기여

버그 리포트나 개선 제안은 이슈로 등록해주세요.
