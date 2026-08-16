# Contract Extractor 프로젝트 가이드

## 📋 프로젝트 개요

**Contract Extractor**는 PDF 형식의 계약서 문서에서 구조화된 정보를 자동으로 추출하는 파이프라인 시스템입니다.

### 주요 목적
- PDF 계약서를 입력받아 텍스트를 추출 (텍스트 PDF 및 OCR 지원)
- LLM 기반 날짜 추출 및 정규식 패턴 매칭을 사용하여 계약 관련 필드 추출
- 추출된 정보를 구조화된 JSON 형식으로 출력
- 각 필드에 대한 신뢰도(confidence) 점수와 근거(evidence) 제공

### 처리 흐름
```
PDF 파일 입력
    ↓
[1단계: Ingest] PDF → TextBlock 리스트 변환
    ├─ 텍스트 PDF: pdfminer.six로 직접 추출
    └─ 스캔 PDF: OCR API 호출 (블록 수가 적을 때 자동 폴백)
    ↓
[2단계: Extract] 각 필드별 추출기 실행 → Candidate 후보 수집
    ├─ 날짜 필드: LLM 기반 추출 (체결일, 시작일, 종료일)
    ├─ 기타 필드: 정규식 및 휴리스틱 기반 추출
    └─ LLM 실패 시 Fallback 로직 사용
    ↓
[3단계: Scoring] 후보들에 점수 부여
    ↓
[4단계: Normalize] 날짜/금액/통화 형식 정규화
    ↓
[5단계: Validate] 최종 후보 선택 및 교차 검증
    ↓
JSON 결과 출력
```

---

## 📁 프로젝트 구조

```
project/
├── app/                          # 메인 애플리케이션 코드
│   ├── __init__.py              # 패키지 초기화
│   ├── extractors/              # 필드별 추출기 모듈
│   ├── ingest/                  # PDF 로딩 및 텍스트 처리
│   │   ├── loader_pdf.py       # PDF 텍스트 추출
│   │   ├── loader_ocr.py       # OCR API 연동
│   │   └── date_llm_extractor.py # LLM 기반 날짜 추출
│   ├── models/                  # 데이터 모델 정의
│   ├── pipeline/                # 파이프라인 오케스트레이션
│   ├── postprocess/             # 후처리 (점수화, 정규화, 검증)
│   └── utils/                   # 공통 유틸리티 함수
├── cli.py                       # CLI 진입점
├── compare_results.py           # 성능 평가 스크립트
├── debug_*.py                   # 디버깅 스크립트들
├── expected/                    # 정답 파일 (ground truth)
├── outputs/json/                # 추출 결과 저장 디렉토리
├── samples/                     # 샘플 PDF 파일
├── requirements.txt             # Python 의존성 목록
└── README.md                    # 기본 README
```

---

## 🔍 각 모듈 상세 설명

### 1. `app/ingest/` - PDF 입력 처리

#### `loader_pdf.py`
**역할**: PDF 파일을 텍스트 블록 리스트로 변환

**주요 기능**:
- `pdfminer.six` 라이브러리를 사용하여 PDF에서 텍스트 추출
- 각 텍스트 라인을 `TextBlock` 객체로 변환
- 페이지 번호와 라인 번호 정보 보존
- OCR 폴백 지원 (블록 수가 적을 때 자동으로 OCR API 호출)

**데이터 구조**:
```python
@dataclass
class TextBlock:
    page: int              # 페이지 번호 (1부터 시작)
    line_no: int           # 라인 번호
    text: str              # 텍스트 내용
    bbox: Optional[Tuple]  # 바운딩 박스 (현재 미사용)
```

**함수**:
- `load_pdf_as_blocks(pdf_path: str) -> List[TextBlock]`: PDF를 TextBlock 리스트로 변환 (텍스트 PDF만)
- `load_pdf_via_ocr(pdf_path, ocr_url, timeout) -> List[TextBlock]`: OCR API를 사용하여 텍스트 추출
- `load_pdf_with_fallback_ocr(pdf_path, ocr_url, min_blocks_threshold, timeout) -> List[TextBlock]`: 텍스트 추출 실패 시 OCR 폴백

**OCR 폴백 로직**:
1. 먼저 `pdfminer.six`로 텍스트 추출 시도
2. 추출된 블록 수가 `min_blocks_threshold` (기본값: 10) 미만이면 OCR API 호출
3. OCR 실패 시 기존 블록 반환 (폴백)

**설계 의도**:
- 텍스트 PDF는 빠르게 직접 처리
- 스캔 PDF는 자동으로 OCR API 호출하여 처리
- 페이지/라인 정보를 보존하여 나중에 근거(evidence) 추적 가능

#### `loader_ocr.py`
**역할**: OCR API를 통한 스캔 PDF/이미지 텍스트 추출

**주요 기능**:
- OCR API 호출하여 PDF/이미지에서 텍스트 추출
- OCR 응답을 `TextBlock` 형식으로 변환하여 기존 파이프라인과 호환
- 환경변수 또는 `.env` 파일에서 API URL 읽기

**함수**:
- `call_ocr_api(file_path, ocr_url, timeout) -> dict`: OCR API 호출
- `ocr_response_to_blocks(ocr_json) -> List[TextBlock]`: OCR 응답을 TextBlock 리스트로 변환
- `load_pdf_via_ocr(pdf_path, ocr_url, timeout) -> List[TextBlock]`: 전체 OCR 로딩 프로세스

**설정**:
- `OCR_API_URL`: OCR API 엔드포인트 URL (환경변수 또는 .env 파일)
- `OCR_TIMEOUT`: 요청 타임아웃 (초, 기본값: 60)

#### `date_llm_extractor.py`
**역할**: LLM을 사용한 날짜 추출 (체결일, 시작일, 종료일)

**주요 기능**:
- 한 번의 LLM 호출로 체결일, 시작일, 종료일을 모두 추출하여 비용 절감
- 효력일/발효일 관련 키워드 인식
- 캐싱으로 동일 문서 재처리 시 비용 절감
- LLM 실패 시 Fallback 로직 사용

**함수**:
- `extract_dates_with_llm(blocks, use_cache) -> Dict[str, Optional[str]]`: LLM 기반 날짜 추출
- `collect_date_related_texts(blocks) -> List[Tuple]`: 날짜 관련 텍스트 수집
- `build_date_prompt(date_texts) -> str`: LLM 프롬프트 구성
- `parse_llm_date_response(response) -> Dict`: LLM 응답 파싱
- `call_bedrock_api(prompt, bedrock_url, timeout) -> dict`: Bedrock API 호출

**설정**:
- `BEDROCK_API_URL`: Bedrock API 엔드포인트 URL (환경변수 또는 .env 파일)
- `BEDROCK_TIMEOUT`: 요청 타임아웃 (초, 기본값: 60)

**캐싱**:
- 동일한 blocks 해시에 대해 캐시 사용
- `clear_cache()` 함수로 캐시 초기화 가능

#### `detector_text.py`
**역할**: 텍스트 PDF 감지 모듈 (향후 확장용, 현재 사용되지 않음)

---

### 2. `app/extractors/` - 필드 추출기

모든 추출기는 `FieldExtractor` 베이스 클래스를 상속받아 `extract()` 메서드를 구현합니다.

#### `base.py`
**역할**: 추출기의 공통 인터페이스와 데이터 구조 정의

**데이터 구조**:
```python
@dataclass
class Evidence:
    page: int              # 발견된 페이지
    lines: List[int]       # 발견된 라인 번호들
    snippet: str           # 발견된 텍스트 스니펫

@dataclass
class Candidate:
    field: str             # 필드명 (예: "CNT_CON_DATE")
    raw_value: str         # 추출된 원시 값
    source: str            # 출처 ("header" | "sentence" | "table" | "title")
    evidence: Evidence     # 근거 정보
    features: Dict         # 특징 딕셔너리 (점수 계산에 사용)
    score: float = 0.0     # 점수 (후처리에서 채워짐)
```

**베이스 클래스**:
```python
class FieldExtractor:
    def extract(self, blocks: List[TextBlock]) -> List[Candidate]:
        raise NotImplementedError
```

#### `cnt_identifier.py` - 계약번호 추출기
**역할**: 계약서에서 계약번호 또는 문서번호 추출

**상태**: 현재 미사용 (주석 처리됨)

**추출 패턴**:
- "계약 번호", "문서 번호", "Contract No.", "Agreement ID" 등의 라벨 뒤의 식별자
- 정규식: `(계약\s*번호|문서\s*번호|Contract\s*No\.?|Agreement\s*ID)\s*[:\-]?\s*([A-Za-z0-9\-_/]+)`

**출력 필드**: `TEMP_KEY`

**참고**: 필요시 `app/pipeline/router.py`의 `REGISTERED_EXTRACTORS`에서 주석 해제하여 사용 가능

#### `cnt_name.py` - 계약명 추출기
**역할**: 계약서의 제목/계약명 추출

**추출 전략**:
- 1페이지 상단 10줄 이내의 텍스트만 검사
- "계약서", "Agreement", "Contract" 등의 키워드 포함 여부 확인
- 길이 제한: 6~60자

**출력 필드**: `CNT_NAME`

#### `cnt_con_date.py` - 계약 체결일 추출기
**역할**: 계약 체결일(서명일) 추출

**추출 전략**:
1. **LLM 기반 추출 (최우선)**:
   - `date_llm_extractor.py`를 사용하여 LLM으로 체결일 추출
   - 효력일/발효일 관련 키워드 인식
   - 추출된 날짜가 포함된 텍스트 블록 찾기 (Evidence용)

2. **Fallback 로직** (LLM 실패 시):
   - 1순위: "계약 체결일", "서명일", "날인일" 등의 라벨이 있는 줄의 날짜
   - 2순위: 마지막 페이지의 하단에서 날짜만 있는 블록
   - 3순위: 마지막 페이지의 하단에서 일반 날짜 블록
   - 4순위: 전체 문서에서 마지막 날짜 (기간 범위 제외)

**특징 점수**:
- `llm_extracted`: LLM 추출 여부 (+1.0)
- `label`: 라벨 존재 여부 (+0.30)
- `last_page`: 마지막 페이지 여부 (+0.25)
- `bottom_zone`: 페이지 하단 10줄 이내 (+0.20)
- `sign_ctx`: 서명/날인 컨텍스트 (+0.20)
- `format`: 날짜 형식 적합 (+0.20)

**출력 필드**: `CNT_CON_DATE`

#### `cnt_period.py` - 계약 기간 추출기
**역할**: 계약 시작일과 종료일 추출

**추출 전략**:
1. **LLM 기반 추출 (최우선)**:
   - `date_llm_extractor.py`를 사용하여 LLM으로 시작일, 종료일 추출
   - 기간 범위 패턴 인식
   - 효력일/발효일 관련 키워드로 시작일 추론
   - 시작일이 없으면 체결일을 시작일로 사용

2. **Fallback 로직** (LLM 실패 시):
   - 날짜 범위 패턴: `2024.01.01 ~ 2024.12.31`
   - 한글 패턴: `2024년 1월 1일부터 2024년 12월 31일까지`
   - 영문 패턴: `from 2024-01-01 to 2024-12-31`

**특징**:
- `llm_extracted`: LLM 추출 여부 (+1.0)
- `header`: "유효기간", "계약기간", "Term" 등의 라벨 존재 여부

**출력 필드**: `CNT_ST_DATE`, `CNT_END_DATE`

#### `cnt_amount.py` - 계약 금액 추출기
**역할**: 계약 금액과 통화 추출

**추출 전략**:
1. **문맥 필터링**:
   - 좋은 토큰: "계약금액", "총액", "대가", "대금", "Contract Value" 등
   - 나쁜 토큰: "인지세", "수수료", "이자", "보증금", "선급금" 등
   - 포함 동사: "으로 한다", "지급한다", "정한다", "산정한다" 등
   - 제외 동사: "부담한다", "공제한다", "납부한다", "환급한다" 등
   - 통화 단서: KRW, USD, EUR, JPY, 원, ₩, $, €, ¥ 등

2. **정규식 매칭**:
   - 금액 패턴: 숫자 + 쉼표/점 구분자
   - 통화 기호: ₩, $, €, ¥ 등
   - 통화 코드: KRW, USD, EUR, JPY 등
   - 여러 금액 후보 중 통화 기호/코드가 붙은 것 우선, 없으면 자릿수가 가장 큰 숫자 우선

3. **필터링 로직**:
   - 나쁜 토큰 + 제외 동사가 있으면 스킵
   - 좋은 신호(좋은 토큰, 포함 동사, 통화 단서)가 하나도 없으면 스킵

**출력 필드**: `CNT_AMT`, `CNT_AMT_CRY`

#### `cnt_auto_renewal.py` - 자동 갱신 기간 추출기
**역할**: 자동 갱신 조건의 기간과 단위 추출

**추출 패턴**:
- 트리거 키워드: "자동 갱신", "자동 연장", "Auto Renewal" 등
- 기간 패턴: "1년", "6개월", "30일" 등

**단위 매핑**:
- 년/year → YEAR
- 개월/월/month → MONTH
- 일/day → DAY

**출력 필드**: `CNT_AUTO_RNW_TERM_AMT`, `CNT_AUTO_RNW_TERM_UNIT`

#### `cnt_renewal_flag.py` - 재계약 여부 추출기
**역할**: 재계약 문서인지 여부 판단 (O/X)

**판단 로직**:
- "재계약", "갱신계약", "Renewal" 등의 키워드 발견 시
- 선행 계약 참조("기체결", "원계약", 계약번호 등)가 있으면 → **O**
- 자동 갱신 문구만 있으면 → **X**

**출력 필드**: `CNT_RENEWAL`

---

### 3. `app/postprocess/` - 후처리 모듈

#### `scoring.py` - 점수화
**역할**: 각 Candidate 후보에 신뢰도 점수 부여

**점수 계산 방식**:
1. **출처 가중치** (source weight):
   - `header`: 0.45 (가장 신뢰도 높음)
   - `sentence`: 0.35
   - `table`: 0.35
   - `title`: 0.20

2. **특징 가중치** (hint weights):
   - `label`: 0.30 (라벨 존재)
   - `format`: 0.20 (형식 적합)
   - `last_page`: 0.25 (마지막 페이지)
   - `bottom_zone`: 0.20 (하단 영역)
   - `sign_ctx`: 0.20 (서명 컨텍스트)
   - `kw`: 0.25 (키워드 매칭)
   - `verb`: 0.20 (동사 매칭)
   - `ccy`: 0.20 (통화 단서)

3. **기타 특징**: 각각 0.1씩 가점

**최종 점수**: 0.0 ~ 1.0 범위로 클리핑

#### `normalize.py` - 정규화
**역할**: 추출된 값들을 표준 형식으로 변환

**정규화 규칙**:
- **날짜 필드** (`CNT_CON_DATE`, `CNT_ST_DATE`, `CNT_END_DATE`):
  - `2024.7.1` → `20240701`
  - `2024-07-01` → `20240701`
  - `2024년 7월 1일` → `20240701`
  - `1 July 2024` → `20240701`
  - `20240701` (이미 8자리) → 그대로 유지

- **금액 필드** (`CNT_AMT`):
  - 쉼표 및 공백 제거: `1,000,000` → `1000000`
  - 정규식: `[,\s]` 패턴으로 콤마와 공백 모두 제거

- **통화 필드** (`CNT_AMT_CRY`):
  - `원`, `₩` → `KRW`
  - `$`, `달러` → `USD`
  - `€` → `EUR`
  - 대문자 변환

- **자동 갱신 단위** (`CNT_AUTO_RNW_TERM_UNIT`):
  - 대문자 변환

#### `validate.py` - 검증 및 최종 선택
**역할**: 후보들 중 최고 점수 후보 선택 및 교차 검증

**처리 단계**:
1. **필드별 그룹화**: 같은 필드의 후보들을 묶음
2. **최고 점수 선택**: 각 필드에서 점수가 가장 높은 후보 선택
3. **교차 검증**:
   - 시작일 > 종료일인 경우 종료일 신뢰도 감소 (-0.3)
   - 금액은 있는데 통화가 없으면 통화 필드에 안내 메시지 추가
4. **임계치 검사**: 점수가 0.75 미만이면 "검수 필요" 플래그 추가

**출력 형식**:
```python
{
    "CNT_CON_DATE": {
        "value": "20240701",
        "confidence": 0.95,
        "evidence": {
            "page": 1,
            "lines": [38],
            "snippet": "2024년 7월 1일"
        },
        "note": null  # 또는 "검수 필요" 등
    },
    ...
}
```

---

### 4. `app/pipeline/` - 파이프라인 오케스트레이션

#### `router.py` - 메인 라우터
**역할**: 전체 파이프라인을 조율하는 핵심 모듈

**처리 흐름**:
1. PDF 로딩: `load_pdf_as_blocks()` 호출
2. 추출기 실행: 모든 등록된 추출기를 순회하며 후보 수집
   - 개별 추출기 오류는 전체 파이프라인을 중단시키지 않음 (방어적 프로그래밍)
3. 후처리 파이프라인:
   - `scoring.apply()`: 점수 부여
   - `normalize.apply()`: 정규화
   - `validate.select_best()`: 최종 선택
4. 결과 구성: 빈 결과 템플릿에 최종 값 채우기
5. 특수 처리: `CNT_CONCLUDED` 필드는 항상 사용자 입력 필요로 표시

**등록된 추출기**:
```python
REGISTERED_EXTRACTORS = [
    # IdentifierExtractor(),   # 계약번호 (현재 미사용)
    NameExtractor(),            # 계약명
    ConDateExtractor(),         # 체결일 (LLM 기반)
    PeriodExtractor(),          # 기간 (LLM 기반)
    AmountExtractor(),          # 금액
    AutoRenewalExtractor(),     # 자동 갱신
    RenewalFlagExtractor(),     # 재계약 여부
]
```

#### `run.py` - 실행 래퍼
**역할**: 간단한 실행 인터페이스 제공

**함수**:
- `run_on_file(pdf_path: str) -> dict`: 단일 파일 처리

#### `config.py` - 설정
**역할**: 점수 가중치와 임계치 설정

**설정 항목**:
- `WEIGHTS`: 출처별 가중치, 형식 검증 가중치 등
- `THRESHOLDS`: 수락 임계치 (0.75)

---

### 5. `app/models/` - 데이터 모델

#### `result.py` - 결과 모델
**역할**: 최종 결과 데이터 구조 정의

**함수**:
- `make_empty_result() -> dict`: 모든 필드가 None인 빈 결과 템플릿 생성

**필드 목록**:
- `TEMP_KEY`: 임시 키 (계약번호)
- `CNT_NAME`: 계약명
- `CNT_CON_DATE`: 계약 체결일
- `CNT_ST_DATE`: 계약 시작일
- `CNT_END_DATE`: 계약 종료일
- `CNT_CONCLUDED`: 체결 여부 (사용자 입력)
- `CNT_RENEWAL`: 재계약 여부 (O/X)
- `CNT_AMT`: 계약 금액
- `CNT_AMT_CRY`: 통화
- `CNT_AUTO_RNW_TERM_AMT`: 자동 갱신 기간 수치
- `CNT_AUTO_RNW_TERM_UNIT`: 자동 갱신 기간 단위

---

### 6. `app/utils/` - 유틸리티

#### `regexes.py` - 정규식 패턴 모음
**역할**: 프로젝트 전반에서 사용하는 정규식 패턴을 중앙 집중 관리

**주요 패턴**:
- `RE_DATE_GENERIC`: 다양한 날짜 형식 매칭
- `RE_RANGE1`, `RE_RANGE2`: 날짜 범위 패턴
- `RE_AMOUNT_LINE`: 금액 + 통화 패턴
- `RE_CON_LABEL`: 체결일 라벨 패턴
- `RE_SIGN_CONTEXT`: 서명 컨텍스트 패턴
- `RE_RENEWAL_WORD`: 재계약 키워드
- `RE_PRIOR_REF`: 선행 계약 참조 패턴
- `RE_AUTO_TRIGGER`: 자동 갱신 트리거 키워드
- `RE_AUTO_PERIOD`: 자동 갱신 기간 패턴

#### `text.py` - 텍스트 처리 유틸리티
**역할**: 텍스트 분석 헬퍼 함수

**함수**:
- `is_title_zone(block: TextBlock) -> bool`: 텍스트 블록이 제목 영역인지 판단
  - 1페이지 상단 10줄 이내
  - 길이 6~60자
  - "계약서", "Agreement", "Contract" 키워드 포함

#### `profile.py` - 프로파일 관리
**역할**: 문서 타입별 프로파일 로딩 및 추정

**상태**: 현재 미사용 (프로파일 파일 없음)

**함수**:
- `load_profiles() -> Dict[str, dict]`: `resources/profiles.yml` 파일 로드 (캐싱)
- `guess_profile(blocks, profiles) -> str`: 문서의 프로파일 추정
  - 1페이지 상단 10줄을 제목으로 가정
  - 프로파일의 `title_good` 토큰과 매칭

**참고**: 현재 `AmountExtractor`는 프로파일을 사용하지 않고 하드코딩된 토큰 리스트를 사용합니다.

---

### 8. 루트 파일

#### `cli.py` - CLI 진입점
**역할**: 명령줄 인터페이스 제공

**사용법**:
```bash
python cli.py samples/sample.pdf
```

**처리 과정**:
1. 명령줄 인자에서 PDF 경로 읽기
2. `run_on_file()` 호출하여 처리
3. 결과를 `outputs/json/` 디렉토리에 JSON 파일로 저장
4. 파일명: `{원본파일명}.json`

#### `requirements.txt` - 의존성 목록
**패키지**:
- `pdfminer.six>=20240706`: PDF 텍스트 추출
- `python-dotenv>=1.0.0`: 환경변수 관리 (.env 파일 지원)
- `requests>=2.31.0`: HTTP 요청 (OCR/LLM API 호출)

---

## 🔄 데이터 흐름 상세

### 1단계: PDF → TextBlock
```
sample.pdf
    ↓ (pdfminer.six)
[
  TextBlock(page=1, line_no=1, text="계약서"),
  TextBlock(page=1, line_no=2, text="..."),
  ...
]
```

### 2단계: TextBlock → Candidate
```
각 추출기가 실행되어 후보 수집:
[
  Candidate(field="CNT_CON_DATE", raw_value="2024년 7월 1일", 
            source="header", evidence=Evidence(...), features={...}),
  Candidate(field="CNT_CON_DATE", raw_value="2024-07-01", 
            source="sentence", evidence=Evidence(...), features={...}),
  ...
]
```

### 3단계: Candidate → Scored Candidate
```
점수 부여:
[
  Candidate(..., score=0.95),  # 라벨 있음 + 마지막 페이지
  Candidate(..., score=0.55),  # 일반 문장
  ...
]
```

### 4단계: Scored Candidate → Normalized Candidate
```
정규화:
[
  Candidate(field="CNT_CON_DATE", raw_value="20240701", ...),
  ...
]
```

### 5단계: Normalized Candidate → Final Result
```
필드별 최고 점수 후보 선택:
{
  "CNT_CON_DATE": {
    "value": "20240701",
    "confidence": 0.95,
    "evidence": {...}
  },
  ...
}
```

---

## 🎯 설계 철학

1. **모듈화**: 각 필드 추출기가 독립적으로 동작
2. **방어적 프로그래밍**: 개별 모듈 오류가 전체 파이프라인을 중단시키지 않음
3. **확장성**: 새로운 추출기는 `FieldExtractor`를 상속받아 쉽게 추가 가능
4. **투명성**: 각 필드에 대해 근거(evidence)와 신뢰도(confidence) 제공
5. **유연성**: 프로파일 시스템으로 다양한 문서 타입 지원

---

## 📝 사용 예시

### 기본 사용
```bash
python cli.py samples/sample.pdf
```

### Python 코드에서 사용
```python
from app.pipeline.run import run_on_file

result = run_on_file("samples/sample.pdf")
print(result["CNT_CON_DATE"]["value"])  # "20240701"
print(result["CNT_CON_DATE"]["confidence"])  # 0.95
```

### 개별 추출기 테스트
```python
from app.ingest.loader_pdf import load_pdf_as_blocks
from app.extractors.cnt_con_date import ConDateExtractor

blocks = load_pdf_as_blocks("samples/sample.pdf")
extractor = ConDateExtractor()
candidates = extractor.extract(blocks)
for cand in candidates:
    print(f"{cand.raw_value}: {cand.score}")
```

---

## 🔧 확장 방법

### 새로운 필드 추출기 추가
1. `app/extractors/`에 새 파일 생성 (예: `cnt_party.py`)
2. `FieldExtractor` 상속
3. `extract()` 메서드 구현
4. `app/pipeline/router.py`의 `REGISTERED_EXTRACTORS`에 추가

### 새로운 프로파일 추가
1. `app/resources/profiles.yml`에 새 프로파일 추가
2. `good_tokens`, `bad_tokens`, `title_good` 정의

### 점수 가중치 조정
1. `app/pipeline/config.py`의 `WEIGHTS` 수정
2. `app/postprocess/scoring.py`의 `HINT_W` 수정

---

## ⚠️ 주의사항

1. **OCR 지원**: 텍스트 추출 실패 시 자동으로 OCR API 호출 (블록 수가 적을 때)
2. **LLM API**: 날짜 추출을 위해 LLM API가 필요하지만, 설정되지 않으면 Fallback 로직 사용
3. **한국어/영문 혼용**: 한국어와 영문 계약서 모두 지원
4. **신뢰도 점수**: 0.75 미만이면 검수 필요로 플래그됨
5. **체결 여부**: `CNT_CONCLUDED`는 항상 사용자 입력 필요 (자동 추출 불가)
6. **프로파일**: 현재 프로파일 기능은 사용되지 않음 (향후 확장 가능)
7. **계약번호**: `IdentifierExtractor`는 현재 미사용 (필요시 주석 해제)

---

이 문서는 프로젝트의 전체 구조와 각 모듈의 역할을 이해하는 데 도움이 됩니다.

