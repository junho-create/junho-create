# combined_e25_28791

`combined_e24_table_refined`(27,320건) + `ocr_filter_result/batch5400/final_gt_1571.jsonl`(공공 OCR 데이터셋 hardcase 정제 결과, 1,571건에서 태그/포맷 위반 100건 제외 후 1,471건)을 합친 버전. 합계 28,791건.

## final_gt_1571 → final_gt_1471 변환/필터링

1. **bbox 재정규화**: 원본은 `bbox_scale=1024`, `prompt_style="chandra_with_ocr"`로 이미 Chandra div-HTML target 형식이었음. `gt_html`의 모든 `data-bbox="x0 y0 x1 y1"`와 `ocr_info[].bbox`를 `round(v*1000/1024)` + `[0,1000]` clamp로 재정규화(`build_chandra_dataset.py`의 `renorm_scale`과 동일 공식). `bbox_scale` 1000으로 갱신. `prompt_style`은 그대로 유지(아래 이유 참고).
2. **태그/포맷 위반 100건 제외** — `train/vlm/utils/prompt_templates.py`의 허용 태그 34종 / 허용 속성 14종 화이트리스트와 대조해 아래에 해당하는 행을 전부 제외:

   | 문제 | 건수 |
   |---|---|
   | `<img src="...">` 값 채움 (스펙: src 채우지 말 것) | 56 |
   | `<p></p>` 등 라벨은 있는데 내용이 완전히 빈 블록 | 22 |
   | `<math>` 안에 KaTeX LaTeX 대신 raw MathML(`<mi>/<mo>/<mfrac>/<msub>...`) | 17 |
   | `<colgroup>/<col>` (허용 태그 아님) | 2 |
   | `<tfoot>` (허용 태그 아님) | 1 |
   | 인라인 `<svg><path>...` (diagram은 mermaid로 표현해야 함) | 1 |
   | CSS `background-image: url('data:image/svg+xml;base64,...')`로 서명 낙서를 base64 SVG로 삽입 | 2 |

   1차 스캔에서 98건, 2차 스캔(첫 필터링 결과를 `data:`/`base64` 패턴으로 재검사)에서 2건 추가 발견 → 총 100건 제외. 1,571 − 100 = 1,471.

3. **남겨둔 residual 이슈(제외 대상 아님)**:
   - `<input type="text" size="N">`의 `size` 속성 23건(전부 1개 행에 몰려있음, `109/1990.../5350108-1990-0001-1769.jpg`) — 허용 속성 목록엔 없지만 입력칸 폭 표시용으로 내용을 왜곡하지 않는 경미한 이슈라 제외하지 않음.
   - `<div>`/`</div>` 개수 불일치 35건, Table div가 단일 `<table>...</table>` 블록이 아닌 경우 38건 — 표 셀 안 인라인 `<div style="...">`가 안 닫힌 채 다음 Table div로 넘어가는 원본 GT 특유의 비정형 HTML. 변환 전 원본 파일에도 동일하게 존재함을 대조 확인(스크립트가 만든 문제 아님). `combined_e24_table_refined`에서 이미 겪었던 것과 같은 종류(원본 데이터 특성).

4. 변환/필터링 최종 결과는 `combined_e24_table_refined/final_gt_1471.jsonl`에 별도 저장(중간 산출물). 원래 있던 `final_gt_1571.jsonl`/`final_gt_1473.jsonl`은 필터링 재작업 과정에서 대체되어 삭제됨.

## 분할

`final_gt_1471.jsonl`을 `data.split_dataset`(count 모드, seed=42)로 train/valid/test = 1223/142/106 분할. 비율은 `combined_e24_table_refined`의 기존 train:valid:test = 22727:2629:1964 (≈83.2:9.6:7.2%) 비율 그대로 적용. `complexity` 필드가 없어 층화추출은 `unknown` 그룹 하나로 처리되어 실질적으로는 랜덤 분할.

이후 각 split을 `combined_e24_table_refined/{train,valid,test}.jsonl` 뒤에 그대로 concat(추가 셔플 없음).

| split | e24_table_refined | +final_gt_1471 | combined_e25_28791 |
|---|---|---|---|
| train | 22,727 | 1,223 | 23,950 |
| valid | 2,629 | 142 | 2,771 |
| test | 1,964 | 106 | 2,070 |
| **합계** | **27,320** | **1,471** | **28,791** |

## 검증 결과 (final_gt_1471 대상, 필터링 후 재검증)

- `image_path` 전체 28,791건 기준 중복 없음(0건). 원본 이미지 파일 1,471개 전부 디스크 존재 확인.
- `ocr_info` 전부 채워져 있음 — 비어있는 행 0건, bbox 이상 0건. `text`/`bbox`만 있고 `score`는 없는 형태로 `data/add_ocr.py`(`normalize_ocr_items(..., keep_score=False)`) 산출물과 동일 스키마 — paddle ocr v5 파이프라인으로 정상 생성된 것으로 보임.
- 허용 태그 화이트리스트 위반 0건(필터링 후), 빈 `<p></p>` 0건, `empty gt_html`/`empty image_path` 0건.
