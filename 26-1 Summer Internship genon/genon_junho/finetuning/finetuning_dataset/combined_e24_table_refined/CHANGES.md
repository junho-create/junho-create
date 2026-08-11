# combined_e24_table_refined

`combined_e24_refined`의 `gt_html` 중 `data-label="Table"` div 내부만 `paddlevl16_infer`의 PaddleOCR-VL 1.6 표 추론 결과로 교체한 버전. 그 외 필드/다른 라벨의 div는 원본 그대로.

## 대상

- `prompt_style="unified_layout"` 행만 해당 (`unified_table_with_ocr`는 표 크롭 이미지라 paddle 추론 대상과 `image_path`가 달라 자동으로 제외됨).
- 같은 `image_path`로 `paddlevl16_infer`에서 `pred_tables`를 찾아 페이지 내 GT table div와 bbox IoU로 1:1 매칭 (IoU < 0.3이면 매칭 안 함 → 원본 유지).

## 교체 조건 (모두 만족해야 교체)

table div 안이 `<table>...</table>` 단일 블록이 아닌 경우(다른 div가 섞인 비정형 표, 164건)는 애초에 교체 후보에서 제외.

1. **글자수 비율** `min(len(GT), len(pred)) / max(len(GT), len(pred)) >= 0.85`
   → 캡션이 표에 같이 묶여 들어가거나 일부만 캡처된 경우 방지
2. **앞부분 15자 유사도 >= 0.6**, **뒷부분 15자 유사도 >= 0.6** (공백 제거 후 비교)
   → 전혀 다른 표로 매칭되거나(엉뚱한 페이지/환각) 표 경계가 어긋난 경우 방지
3. **전체 텍스트 유사도(`difflib.SequenceMatcher`) >= 0.5**
   → OCR 오타·LaTeX 표기 차이(`\dagger` vs `†` 등) 같은 잡음은 통과시키되, 중간 내용이 통째로 다른 경우는 차단

텍스트 비교 시 paddle 쪽 셀 안에 리터럴 `\n`(줄바꿈이 문자열 "\n"으로 들어간 것)은 공백으로 치환 후 비교했고, 실제로 교체해 넣는 html에도 동일하게 치환해서 넣음.

## 결과

| split | table div 총계 | paddle 매칭 후보 | 교체됨 |
|---|---|---|---|
| train | 5,519 | 5,152 | 4,317 |
| valid | 620 | 572 | 491 |
| test | 604 | 560 | 470 |
| **합계** | **6,743** | **6,284** | **5,278** |

- 교체된 표가 있는 행: train 2,847 / valid 337 / test 318 (총 3,502행)
- `train.jsonl`, `valid.jsonl`, `test.jsonl` 라인 수는 원본과 동일 (교체 안 된 행은 원본 그대로 복사).
