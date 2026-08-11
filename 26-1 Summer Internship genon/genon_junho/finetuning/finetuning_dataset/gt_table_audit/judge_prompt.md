첫 번째 이미지는 **원본 문서 페이지**이고, 이 페이지에서 표로 라벨링된 영역이
빨간 사각형과 `#1`, `#2` … 번호로 표시되어 있습니다.

두 번째 이미지는 그 표들의 **GT HTML을 브라우저에서 렌더링한 결과**이며, 같은
`#1`, `#2` … 번호가 붙어 있습니다.

당신의 역할은 GT HTML이 원본 페이지의 표를 얼마나 정확하게 담고 있는지 표 단위로
평가하는 것입니다. 이 GT는 학습 데이터로 쓰이므로, 조금이라도 어긋나면 감점해야 합니다.

## 중요 규칙

- **번호가 같은 표끼리만** 비교하세요. `#1`은 `#1`과, `#2`는 `#2`와 비교합니다.
- 첫 번째 이미지에서 빨간 박스 **바깥**에 있는 본문, 제목, 그림, 머리말/꼬리말은
  평가 대상이 아닙니다. 무시하세요.
- 반드시 이미지에 실제로 보이는 내용만 근거로 판단하세요. 추측하지 마세요.
- 반드시 JSON 객체 하나만 출력하세요.
- 설명문, 마크다운, 코드블록, 주석, 앞뒤 문장 없이 JSON만 출력하세요.
- 스키마에 없는 필드는 절대 추가하지 마세요.
- `tables` 배열의 길이는 반드시 **표 개수({{N_TABLES}}개)** 와 같아야 하고,
  `index`는 1부터 {{N_TABLES}}까지 순서대로 하나씩 있어야 합니다.
- 모든 문자열 필드는 빈 문자열 허용 가능하나 null은 절대 사용하지 마세요.
- 값이 없으면 null이 아니라 `""` 또는 `[]`를 사용하세요.
- `score`는 반드시 1~5 정수만 사용하세요.
- `confidence`는 0.0 이상 1.0 이하 숫자로 출력하세요.
- bbox, 좌표, 추정 위치 정보는 출력하지 마세요.
- 확인이 어려운 항목은 보수적으로 감점하고, 그 사유를 `reason`에 명시하세요.

## 평가 항목 정의

각 표에 대해 아래 6개 항목을 매깁니다.

### 1. table_coverage
- 표의 시작과 끝 범위가 원본과 동일한가
- 잘리거나 누락된 행/열/블록이 없는가
- 원본에 없는 불필요한 표 요소가 추가되지 않았는가
- **빨간 박스가 표를 제대로 감싸고 있는가** — 표의 일부만 덮거나, 표가 아닌 본문·
  그림을 포함하거나, 표 여러 개를 하나로 묶었으면 `bbox_error`로 감점

### 2. basic_structure
- 행(row) 수와 열(column) 수가 동일한가
- 헤더와 본문 구분이 올바른가
- 셀 분할 기준이 원본과 동일한가

### 3. complex_structure
- rowspan, colspan이 원본과 동일한가
- 병합 셀의 위치와 범위가 정확한가
- 다단 헤더, 그룹 헤더, 중첩 구조가 유지되었는가

### 4. cell_correspondence
- 원본의 각 셀이 결과에서도 올바른 위치의 셀로 대응되는가
- 셀이 과도하게 분할되거나 잘못 합쳐지지 않았는가
- 인접 셀 관계와 행/열 순서가 유지되는가

### 5. text_accuracy
- 각 셀의 텍스트가 정확한가
- 빠진 글자, 오타, 잘못 인식된 문자가 없는가
- 숫자, 소수점, 자릿수, 날짜, 코드값, 단위가 정확한가

### 6. auxiliary_visual_fidelity
- 캡션, 주석, 단위, 각주가 유지되었는가
- 셀 정렬, 줄바꿈, 열 너비, 행 높이, 테두리 구분이 원본과 유사한가
- 사람이 보기에 동일한 표로 인식될 정도의 시각적 유사성이 있는가

## 점수 기준

- 5 = 원본과 거의 완전히 일치
- 4 = 사소한 차이는 있으나 의미 왜곡 없음
- 3 = 일부 차이가 있으며 부분 수정 필요
- 2 = 구조 또는 내용 오류가 뚜렷함
- 1 = 원본 표를 제대로 재현하지 못함

## severity enum

반드시 아래 중 하나만 사용하세요:
- `"NONE"` — 오류 없음
- `"MINOR"` — 사소한 차이 (의미 영향 없음)
- `"MAJOR"` — 유의미한 오류 (부분 수정 필요)
- `"CRITICAL"` — 심각한 오류 (재작업 필요)

## error_types enum

`error_types` 배열 원소는 반드시 아래 값만 사용하세요 (중복 없이):
- `"missing"` — 누락
- `"extra"` — 불필요한 추가
- `"merge_error"` — 병합 오류
- `"split_error"` — 분할 오류
- `"header_error"` — 헤더 구분 오류
- `"text_error"` — 텍스트 오류
- `"numeric_error"` — 숫자/소수점 오류
- `"layout_error"` — 레이아웃/정렬 오류
- `"caption_note_error"` — 캡션/주석/각주 오류
- `"bbox_error"` — 표 영역(빨간 박스) 자체가 잘못 잡힘
- `"not_a_table"` — 빨간 박스 안이 애초에 표가 아님 (본문·그림·양식 등)
- `"uncertain_visibility"` — 이미지 품질로 인해 확인 불확실

## 미라벨 표

첫 번째 이미지에 **명백히 표인데 빨간 박스가 전혀 없는** 영역이 있으면
`unlabeled_tables`에 기록하세요. 확실하지 않으면 넣지 마세요.

## 출력 JSON 스키마

`tables`는 표 개수만큼 원소를 갖습니다. 아래는 표가 1개일 때의 예시입니다.

```json
{
  "schema_version": "gt_table_audit_v1",
  "tables": [
    {
      "index": 1,
      "metrics": {
        "table_coverage":           {"score": 5, "severity": "NONE", "error_types": [], "reason": ""},
        "basic_structure":          {"score": 5, "severity": "NONE", "error_types": [], "reason": ""},
        "complex_structure":        {"score": 5, "severity": "NONE", "error_types": [], "reason": ""},
        "cell_correspondence":      {"score": 5, "severity": "NONE", "error_types": [], "reason": ""},
        "text_accuracy":            {"score": 5, "severity": "NONE", "error_types": [], "reason": ""},
        "auxiliary_visual_fidelity": {"score": 5, "severity": "NONE", "error_types": [], "reason": ""}
      },
      "summary": ""
    }
  ],
  "unlabeled_tables": [],
  "overall": {"confidence": 0.9, "summary": ""}
}
```

`unlabeled_tables` 원소 형식: `{"where": "<페이지 내 위치를 한 문장으로>", "severity": "MAJOR"}`

## 출력 제약

- `schema_version` 값은 `"gt_table_audit_v1"` 그대로 사용하세요.
- `metrics` 아래 6개 항목 이름은 절대 변경하지 마세요.
- `reason`은 한국어 한 문장으로 쓰되, 기술 용어(rowspan, colspan, header 등)는 영어를 쓰세요.
- 오류가 전혀 없는 항목은 `score` 5, `severity` `"NONE"`, `error_types` `[]`,
  `reason` `""`로 두세요.
- `error_types`는 중복 없이 작성하세요.
- `summary`는 그 표의 문제를 한 문장으로 요약합니다. 문제가 없으면 `""`.
- `overall.confidence`는 이미지에서 직접 확인 가능한 정도를 반영하세요.
- JSON 외 다른 텍스트는 절대 출력하지 마세요.
