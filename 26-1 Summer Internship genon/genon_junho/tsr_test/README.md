# dots.ocr vs Chandra div-HTML Table/Layout 실험

이 브랜치는 `dots.ocr` 모델과 Chandra OCR layout 프롬프트를 사용해 `div data-bbox/data-label`가 붙은 HTML을 출력하도록 학습한 모델을 비교·재현하기 위한 코드와 실행 절차를 정리한 브랜치다.

목표는 다음 두 가지다.

1. 문서 전체 이미지에서 레이아웃 영역과 표 영역을 함께 뽑는 모델이 `dots.ocr` 대비 어떤 장단점을 가지는지 확인한다.
2. 표 crop 평가셋에서 기존 TSR/e18 계열 모델, `div+bbox` table-only 모델, table+layout 통합 모델을 같은 조건으로 비교한다.

대용량 학습 데이터, 모델 checkpoint, 평가 결과는 Git에 올리지 않는다. 필요한 파일은 아래에 적은 b200 서버 경로를 사용한다.

## 1. 실험 요약

### 1.1 평가 데이터셋

| 평가 데이터셋 | 설명 | b200 서버 경로 | 사용처 |
|---|---|---|---|
| dp bench 평가 데이터셋 500장 | Genos 문서 이미지 500장으로 구성된 문서 레이아웃/표 평가셋. manifest는 OCR 포함/미포함 버전이 각각 500라인이다. | `shkim@DCTN-0414131307:/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/eval_data/500_genos_val_set` | `dots_ocr`, `chandra_layout`, `chandra_tablelayout` 문서 단위 비교 |
| tsr 모델 평가 데이터셋 200장 | 기존 TSR 모델 평가에 쓰던 표 crop test set 200장. | `shkim@DCTN-0414131307:/NHNHOME/WORKSPACE/0426030039_A/shkim/tsr_test/train/vlm/_train_data/20260317_4_6000/data_split/test.jsonl` | 표 crop 벤치 |

두 세트의 역할이 다르다. 500장은 표 검출까지 포함한 **문서 파이프라인 종합
평가**이고, 200장은 학습 6,000장과 같은 소스의 held-out이라 **표 구조 인식
심화 평가**다 — 학습에 참여하지 않은 모델(dots.ocr 등)과의 절대 비교에는
in-distribution 편향을 감안해야 한다.

### 1.2 dp bench 평가 데이터셋 500장: 문서 레이아웃 + 테이블 비교

비교 모델은 아래 세 개다.

| 모델 | 의미 | 학습 데이터 |
|---|---|---|
| `dots_ocr` | 공개 dots.ocr 계열 기준선 | 별도 학습 없음 |
| `chandra_layout` | 순정 Qwen3.5-9B에 Chandra div-HTML 출력 형식으로 SFT한 layout-only 모델 | 레이아웃 데이터 |
| `chandra_tablelayout` | 순정 Qwen3.5-9B에 Chandra div-HTML 출력 형식으로 SFT한 table+layout 통합 모델 | 표 + 레이아웃 통합 데이터 |

외부 공개모델 비교군 4종 (2026-07 추가 — 추론 산출물 기준, 이 브랜치 파이프라인으로 평가):

| 모델 | 의미 | 학습 데이터 |
|---|---|---|
| `mineru_native` | MinerU2.5-Pro-1.2B base, vLLM 직접 호출(mineru_vl_utils) | 별도 학습 없음 |
| `mineru_hybrid` | 같은 base, mineru 공식 CLI hybrid 백엔드 (native와 별개 run — 요소 분할·후처리 상이) | 별도 학습 없음 |
| `mineru_lora` | MinerU2.5-Pro + 자체 LoRA(r64) 파인튜닝, native 백엔드 | mineru 네이티브 변환 20k 샘플 |
| `paddle` | PaddleOCR-VL-1.6 0.9B + PP-DocLayoutV3 레이아웃 | 별도 학습 없음 |

`chandra_tablelayout`은 `output/chandra_all_e18align_20260623/student_sft` 계열 checkpoint를 사용한다. 같은 table+layout 모델을 dp bench 평가 데이터셋 500장과 tsr 모델 평가 데이터셋 200장에 모두 평가했다.

평가 지표:

| 지표 | 의미 |
|---|---|
| `NID` | 표·figure·chart 내부 텍스트를 제외한 비테이블 본문/레이아웃 텍스트 정렬 정확도. 높을수록 좋다. |
| `TEDS zero(b)` | GT 표와 예측 표를 bbox IoU 기준으로 매칭한 뒤 TEDS를 계산한다. 매칭되지 않은 GT 표는 0점 처리하므로 표 검출률까지 반영한다. |
| `TEDS-S zero(b)` | `TEDS zero(b)`의 구조-only 버전. 셀 텍스트보다 행/열/병합 구조를 본다. |
| `TEDS skip(a)` | 매칭되지 않은 GT 표를 제외하고 TEDS를 계산한다. 검출된 표의 HTML 품질을 보는 지표다. |
| `TEDS-S skip(a)` | `TEDS skip(a)`의 구조-only 버전. |
| `matched/GT` | GT 표 305개 중 bbox IoU 기준으로 예측 표와 매칭된 표 개수. |
| `extra_pred` | GT와 매칭되지 않은 추가 예측 표 개수. |
| `text/RO Edit` | OmniDocBench v1.7의 텍스트/읽기순서 정규화 편집거리. 낮을수록 좋다. |
| `CDM` | 수식을 실제 렌더링해 문자 단위로 비교하는 metric. LaTeX 문법 차이에 강건하다. 높을수록 좋다. |
| `LLM-Judge` | pdf-parse-bench 방식의 LLM 채점(0~10, judge=사내 Qwen3.5-397B). 셀 내용·헤더 매핑 오류를 벌점한다. |

성능:

| 모델 | NID | TEDS zero(b) | TEDS-S zero(b) | TEDS skip(a) | TEDS-S skip(a) | matched/GT | extra_pred |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dots_ocr` | 0.8836 | 0.7923 | 0.8020 | 0.8219 | 0.8320 | 294/305 | 16 |
| `chandra_layout` | 0.8903 | 0.6949 | 0.7175 | 0.7679 | 0.7929 | 276/305 | 34 |
| `chandra_tablelayout` | 0.8905 | 0.6999 | 0.7225 | 0.7936 | 0.8192 | 269/305 | 30 |
| `mineru_native` | 0.9353 | 0.7259 | 0.7484 | 0.7796 | 0.8037 | 284/305 | 32 |
| `mineru_hybrid` | 0.9026 | 0.7100 | 0.7300 | 0.7874 | 0.8096 | 275/305 | 19 |
| `mineru_lora` | 0.9305 | 0.7251 | 0.7482 | 0.7760 | 0.8007 | 285/305 | 20 |
| `paddle` | 0.9019 | 0.6897 | 0.7129 | 0.7763 | 0.8024 | 271/305 | 27 |

신규 metric 성능 (이슈 doc_parser#318에서 도입, `dots_ocr`는 위와 동일 산출물):

| 모델 | text Edit↓ | RO Edit↓ | 수식 Edit↓ | 수식 CDM↑ | TEDS(omni)↑ | LLM 수식/10 (매칭) | LLM 표/10 (매칭) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dots_ocr` | 0.0847 | 0.1255 | 0.0072 | 0.9961 | 0.9441 | 9.97 (30/30) | 5.75 (303/305) |
| `chandra_layout` | 0.0516 | 0.1748 | 0.2022 | 0.8746 | 0.8142 | 8.82 (27/30) | 5.63 (292/305) |
| `chandra_tablelayout` | 0.0666 | 0.1915 | 0.2276 | 0.8426 | 0.8211 | 9.20 (25/30) | 5.66 (285/305) |
| `mineru_native` | 0.0375 | 0.1288 | 0.1256 | 0.8785 | 0.8603 | 9.20 (29/30) | 5.58 (294/305) |
| `mineru_hybrid` | 0.0603 | 0.1765 | 0.1543 | 0.8432 | 0.8381 | 9.40 (30/30) | 5.50 (293/305) |
| `mineru_lora` | 0.0364 | 0.1329 | 0.1504 | 0.8771 | 0.8607 | 9.43 (29/30) | 5.35 (293/305) |
| `paddle` | 0.0672 | 0.1611 | 0.1754 | 0.8397 | 0.8078 | 8.47 (29/30) | 5.26 (287/305) |

LLM 점수는 매칭 성공분 평균이고 괄호가 LLM 매칭 수다(미매칭 원인은 대부분
추출 단계의 모델 폭주 출력 — PHASE3_REPORT §3). TEDS(omni)의 매칭은 텍스트
기반이라 별도(bbox 매칭 수는 위 3.8 성능표의 `matched/GT`).

핵심 해석:

- NID는 모델 간 큰 차이가 작지만, 문서별 실패 양상은 다르다.
- TEDS는 `dots_ocr`이 더 안정적이다.
- Chandra 계열 모델의 주요 약점은 “검출된 표 HTML 품질”보다 “표를 표로 잡는 검출률/표 개수 분리” 쪽에 있다.
- 표 데이터를 섞으면 잡힌 표의 HTML 품질은 좋아질 수 있지만, 문서 전체에서 표 영역을 안정적으로 분리하는 문제는 별도 개선이 필요하다.
- LLM-Judge 기준으로도 위 해석이 성립한다. 검출 성공분만 내용 채점하면 dots 5.75 vs chandra 5.63~5.66으로 동급이고, dots 우위의 실체는 검출력(매칭 294 vs 269~276)이다.
- TEDS 절대값은 후한 방향의 편향이 있다. 검출 실패 표 제외(skip)와 다단(colspan) 헤더 붕괴에 대한 부분점수 때문에 LLM-Judge 대비 ≈0.25 높게 나온다(전 모델 공통).
- 외부 공개모델 4종은 **본문 텍스트·읽기순서에서 기존 전 모델을 상회**(NID 0.90~0.94, text Edit 0.036~0.067 — 특히 mineru 계열)하지만, **표는 검출(matched 271~285)·인식(TEDS omni 0.81~0.86, LLM 5.26~5.58) 모두 dots 우위**가 유지된다. 수식은 CDM 0.84~0.88로 dots(0.996)에 미달.

관련 서버 산출물:

```text
train/vlm/eval_results/genos500_ocr_layout
train/vlm/eval_results/genos500_ocr_tablelayout_epoch3
```

### 1.3 tsr 모델 평가 데이터셋 200장: 표 crop 벤치

| 모델 | 의미 | 가장 높은 성능의 checkpoint |
|---|---|---|
| `e18_6910` | 기존 e18 원본 계열 기준 모델 | ckpt-500 |
| `e18_nodiv` | div 없는 e18 원본 데이터 control 재학습 | ckpt-400 |
| `table` | Chandra div-HTML table-only 모델 | ckpt-200 |
| `tablelayout` | Chandra div-HTML table+layout 통합 모델 | ckpt-1050 |
| `dots_ocr` | dots.ocr layout JSON prompt 기준선 | single run |

대표 수치:

| 모델 | TEDS | TEDS-S | Span F1 | Attr Acc | LLM-Judge/10 |
|---|---:|---:|---:|---:|---:|
| `e18_6910 ckpt-500` | 0.9484 | 0.9621 | 0.8005 | 0.8539 | - |
| `e18_nodiv ckpt-400` | 0.9430 | 0.9606 | 0.7677 | 0.8247 | 7.08 |
| `table ckpt-200` | 0.9384 | 0.9551 | 0.7293 | 0.7879 | 6.91 |
| `tablelayout ckpt-1050` | 0.9439 | 0.9646 | 0.7775 | 0.8293 | 6.81 |
| `dots_ocr` | 0.8254 | 0.8530 | 0.4588 | 0.5290 | 4.73 |
| `mineru_hybrid` | 0.9417 | 0.9682 | 0.8595 | 0.8879 | 6.42 |
| `mineru_lora` | 0.9182 | 0.9434 | 0.8425 | 0.8688 | 6.14 |
| `paddle` | 0.9181 | 0.9613 | 0.8549 | 0.8872 | 6.00 |
| `mineru_native` | 0.9000 | 0.9298 | 0.8234 | 0.8600 | 5.92 |

LLM-Judge(0~10, 이슈 doc_parser#318에서 도입)에서도 같은 서열이 재현된다.
dots 저점 124건 중 65%가 병합/헤더 구조의 실질 실패(표기 규약성 벌점 0건)로
Span F1 격차와 정합한다. 단 TEDS ≥ 0.9인데 LLM ≤ 4인 표가 run당 35~49건
존재하므로(헤더 수식 오독 등), 고TEDS가 곧 정답은 아니다. complexity별 상세는
§3.9 참고.

외부 공개모델 4종(하단 4행, TEDS 계열은 jhyeo측 eval 산출물 기준)은 dots를
크게 상회하고 파인튜닝 3종에는 미달한다. `mineru_hybrid`는 TEDS(0.9417)로는
파인튜닝급이지만 LLM(6.42 vs 6.8~7.1)이 낮아 셀 내용 오류가 상대적으로 많고,
같은 base인 `mineru_native`(0.9000/5.92)와의 격차는 CLI hybrid 백엔드의 표
후처리 효과다. 전체 수치·해석은 EVAL_RESULTS.md §6.

관련 서버 산출물:

```text
train/vlm/eval_results/dots_ocr_6000test
train/vlm/eval_results/table_6000test
train/vlm/eval_results/e18_nodiv_6000test
train/vlm/eval_results/tablelayout_6000test
```

## 2. 서버 기준 경로

작업 루트:

```bash
cd /NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm
```

공통 Python venv:

```bash
source /NHNHOME/WORKSPACE/0426030039_A/shkim/tsr_test/train/vlm/.venv/bin/activate
```

베이스 모델:

```text
/NHNHOME/WORKSPACE/0426030039_A/shkim/models/Qwen3.5-9B
```

Git에 올리지 않는 주요 서버 경로:

| 종류 | b200 서버 경로 |
|---|---|
| 최종 학습 데이터 | `/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/_train_data/chandra_table_layout_divhtml_16886` |
| 최종 학습 이미지 | `/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/_train_data/chandra_table_layout_divhtml_16886/images` |
| 레이아웃 원본 JSONL | `/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/_train_data/layout_src_9984/labeler_converter_layout_source_9984.jsonl` |
| 테이블 원본/분할 데이터 | `/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/_train_data/table_src_6902` |
| table bbox manifest | `/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/_train_data/table_bbox_manifest.json` |
| dp bench 평가 데이터셋 500장 | `/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/eval_data/500_genos_val_set` |
| tsr 모델 평가 데이터셋 200장 | `/NHNHOME/WORKSPACE/0426030039_A/shkim/tsr_test/train/vlm/_train_data/20260317_4_6000/data_split/test.jsonl` |
| 모델 output(LoRA/checkpoint) | `/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/output` |
| 평가 결과 | `/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/eval_results` |

`table_bbox_manifest.json`은 table crop 이미지별 bbox lookup 파일이다. h100 서버의 labeler converter output에서 표 영역 bbox와 이미지 크기를 뽑아 만든 manifest이며, `build_chandra_dataset` 단계에서 표 HTML을 `<div data-bbox="..." data-label="Table">...</div>`로 감쌀 때 사용한다. 학습 샘플 JSONL 자체는 아니고, 표 HTML 데이터에 Chandra layout bbox wrapper를 붙이기 위한 보조 metadata다.

현재 b200 서버에 유지한 주요 모델 output(LoRA/checkpoint):

| output | 의미 | dp bench 500장 성능표 모델명 | tsr 200장 성능표 모델명 | 학습 데이터/설정 |
|---|---|---|---|---|
| `output/chandra_all_e18align_20260623` | 최종 table+layout div-HTML 모델. 두 평가에 공통 사용한 핵심 모델이다. | `chandra_tablelayout` (`genos500_ocr_tablelayout_epoch3`, 본 README 성능표 재현 기준) | `tablelayout` (`tablelayout_6000test`, 대표 ckpt-1050) | `config/exp_chandra_all_e18align.yaml` 기반. `chandra_table_layout_divhtml_16886/data/train.jsonl` 사용: table 6,000장 + layout 7,988장. e18 계열과 하이퍼파라미터를 맞춘 설정. |
| `output/chandra_layout_20260622` | layout-only div-HTML 모델. 표 단독 crop 데이터를 섞지 않았을 때의 문서 레이아웃 모델이다. | `chandra_layout` (`genos500_ocr_layout`) | 미사용 | `config/exp_chandra_layout.yaml` 기반. `chandra_table_layout_divhtml_16886/data/train_layout.jsonl` 사용: layout 7,988장만 학습. |
| `output/chandra_table_e18align_20260623` | table-only div-HTML 모델. 표 crop만 보고 `<div data-bbox ... data-label="Table"><table>...</table></div>`를 출력하도록 학습한 모델이다. | 미사용 | `table` (`table_6000test`, 대표 ckpt-200) | `config/exp_chandra_table_e18align.yaml` 기반. `chandra_table_layout_divhtml_16886/data/train_table.jsonl` 사용: e18 공통 stem 기준 table 6,000장. |
| `output/e18_nodiv_20260624` | div 없는 e18 원본 데이터 control 모델. Chandra div/bbox wrapper를 제거한 조건에서 재학습한 비교군이다. | 미사용 | `e18_nodiv` (`e18_nodiv_6000test`, 대표 ckpt-400) | `config/exp_e18_nodiv_20260624.yaml` 기반. 기존 e18 계열 표 데이터 형식을 유지하고 div wrapper 없이 학습. |

## 3. 학습 및 평가 파이프라인

### 3.1 학습 데이터 구성

최종 학습 데이터는 b200 서버의 아래 경로를 기준으로 한다.

```text
/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/_train_data/chandra_table_layout_divhtml_16886
```

디렉터리명은 `Chandra 프롬프트 + Table/Layout 통합 + div-HTML target + 총 16,886개 샘플`을 의미한다.

이미지 경로:

```text
/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/_train_data/chandra_table_layout_divhtml_16886/images
```

split별 구성은 다음과 같다.

| Dataset | Table | Layout | Table+Layout |
|---|---:|---:|---:|
| Train | 6000 | 7988 | 13988 |
| Valid | 652 | 998 | 1650 |
| Test | 250 | 998 | 1248 |
| 합계 | 6902 | 9984 | 16886 |

주요 파일:

| 파일 | 역할 |
|---|---|
| `data/train.jsonl` | table+layout 통합 train |
| `data/valid.jsonl` | table+layout 통합 valid |
| `data/test.jsonl` | table+layout 통합 test |
| `data/train_table.jsonl`, `data/valid_table.jsonl`, `data/test_table.jsonl` | table-only 실험용 split |
| `data/train_layout.jsonl`, `data/valid_layout.jsonl`, `data/test_layout.jsonl` | layout-only 실험용 split |

최종 target 형식은 Chandra OCR layout prompt가 요구하는 HTML block이다.

```html
<div data-bbox="x0 y0 x1 y1" data-label="Table">
<table>...</table>
</div>
```

공통 규칙:

- `data-bbox`는 `0~1000` 정규화 좌표다.
- `data-label`은 `Caption`, `Footnote`, `Formula`, `List-item`, `Page-footer`, `Page-header`, `Picture`, `Section-header`, `Table`, `Text`, `Title` 중 하나다.
- `Table` content는 `<table>...</table>` HTML이다.
- non-table layout element는 같은 `<div data-bbox ... data-label ...>` wrapper로 감싼다.
- 학습 시 OCR 입력은 샘플 단위로 50% mix된다.

### 3.2 사용 프롬프트

프롬프트는 `train/vlm/utils/prompt_templates.py` 기준이다. 학습과 평가는 같은 system/user prompt를 사용한다.

System prompt:

```text
You are an expert document OCR and layout analysis model. You convert a document image into HTML layout blocks. Output only the HTML layout blocks, with no extra commentary and no markdown code fences.
```

User prompt (`chandra_no_ocr`):

```text
OCR this image to HTML, arranged as layout blocks.  Each layout block should be a div with the data-bbox attribute representing the bounding box of the block in x0 y0 x1 y1 format.  Bboxes are normalized 0-1000. The data-label attribute is the label for the block.

Use the following labels:
- Caption
- Footnote
- Formula
- List-item
- Page-footer
- Page-header
- Picture
- Section-header
- Table
- Text
- Title

Only use these tags ['math', 'br', 'i', 'b', 'u', 'del', 'sup', 'sub', 'table', 'tr', 'td', 'p', 'th', 'div', 'pre', 'h1', 'h2', 'h3', 'h4', 'h5', 'ul', 'ol', 'li', 'input', 'a', 'span', 'img', 'hr', 'tbody', 'small', 'caption', 'strong', 'thead', 'big', 'code', 'chem'], and these attributes ['class', 'colspan', 'rowspan', 'display', 'checked', 'type', 'border', 'value', 'style', 'href', 'alt', 'align', 'data-bbox', 'data-label'].

Guidelines:
* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property. Describe in detail inside the div tag. Also convert charts to high fidelity data, and convert diagrams to mermaid.
* Forms: Mark checkboxes and radio buttons properly.
* Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
* Chemistry: Use <chem>...</chem> tags for chemical formulas with reactive SMILES.
* Lists: Preserve indents and proper list markers.
* Use the simplest possible HTML structure that accurately represents the content of the block.
* Make sure the text is accurate and easy for a human to read and interpret.  Reading order should be correct and natural.
```

User prompt (`chandra_with_ocr`)는 위 `chandra_no_ocr` prompt 뒤에 아래 블록이 추가된다.

```text
In addition to the image, you are provided with OCR text extracted from the same image.
Each OCR item includes recognized text and its bounding box, normalized to the same 0-1000 coordinate space.

<ocr_info>
{ocr_info}
</ocr_info>

Use the OCR text as the primary source of textual content, and use the image to resolve
the layout, region boundaries, labels, merged cells, reading order, and ambiguous cases.
```

### 3.3 원본 데이터 생성 흐름

현재 b200 서버에는 최종 학습 데이터가 이미 만들어져 있으므로 보통 이 단계를 다시 실행하지 않는다. 처음부터 재생성해야 할 때의 흐름은 아래와 같다.

1. 표 이미지 쪽

   - 1-1. 표 이미지를 h100 labeler converter에 통과시켜 Table 영역 bbox를 뽑는다.
   - 1-2. labeler bbox 결과를 `table_bbox_manifest.json`으로 정리한다.
   - 1-3. 기존 표 HTML/OCR 데이터를 실험 split 기준 6,902개로 필터링해 `table_src_6902`를 만든다.

2. 레이아웃 이미지 쪽

   - 2-1. 문서/page 이미지를 h100 또는 기존 labeler converter pipeline에 통과시켜 layout element JSON을 만든다.
   - 2-2. 각 element는 `bbox`, `category`, `text`를 가진다.
   - 2-3. 이 결과를 `layout_src_9984/labeler_converter_layout_source_9984.jsonl`로 정리한다.

3. `table_src_6902`와 `layout_src_9984`를 합쳐 table+layout 공통 중간 JSONL을 만든다.

   table sample은 기존 table HTML을 가진 JSON element 형태로 들어간다. 이때 bbox는 아직 실제 Table bbox가 아니라 전체 영역 placeholder다.

   ```json
   [{"bbox": [0, 0, 1024, 1024], "category": "Table", "text": "<table>...</table>"}]
   ```

   layout sample은 `layout_src_9984`의 bbox/category/text element를 그대로 공통 schema에 맞춰 넣는다.

   ```json
   [{"bbox": [0, 0, 100, 100], "category": "Text", "text": "..."}]
   ```

4. `build_chandra_dataset` 단계에서 최종 Chandra div-HTML target으로 변환한다.

   table은 아래 순서로 변환한다.

   - `table_src` 쪽 기존 table HTML을 꺼낸다.
   - `table_bbox_manifest.json`의 bbox를 가져와 0~1000 좌표계로 정규화한다.
   - category는 `"Table"`로 고정한다.
   - 최종적으로 아래 형태로 감싼다.

   ```html
   <div data-bbox="..." data-label="Table">
   <table>...</table>
   </div>
   ```

   layout은 아래 순서로 변환한다.

   - `layout_src`에서 온 각 element의 bbox/category/text를 사용한다.
   - bbox는 1024 기준에서 1000 기준으로 재정규화한다.
   - category는 원래 layout element category를 유지한다.
   - text는 category에 맞는 HTML content로 바꾼다.
   - 최종적으로 각 layout element를 아래 형태로 감싼다.

   ```html
   <div data-bbox="..." data-label="Text">
   <p>...</p>
   </div>
   ```

표 labeler converter 관련 경로:

| 항목 | 경로 |
|---|---|
| h100 서버 labeler 실행 루트 | `/home/vlm_train/labeler_feat` |
| h100 서버 table labeler output 기록 경로 | `/home/vlm_train/labeler_feat/output/layout_table_7489` |
| b200에 남아 있는 bbox manifest | `/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/_train_data/table_bbox_manifest.json` |
| 현재 학습에 쓰는 filtered table source | `/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/_train_data/table_src_6902` |

`table_bbox_manifest.json`에는 h100 서버 labeler output에서 추출한 표 bbox, 이미지 크기, 원본 PNG 경로가 들어 있다. key는 표 이미지 stem이고, value는 해당 이미지에서 labeler가 찾은 Table 영역 정보다. 예시는 다음 형태다.

`table_src_6902`는 labeler converter raw JSON이 아니다. 실제 학습에 쓰는 표 이미지, 정제된 `gt_html`, 원본 `original_gt_html`, `ocr_info`, `complexity`, `prompt_style` 등을 담은 filtered table source다. 표 bbox는 `table_src_6902`가 아니라 `table_bbox_manifest.json`에서 가져와 최종 div-HTML target을 만들 때 결합한다.

```json
{
  "T04_C06_50011_1421_063": {
    "bbox": [33, 31, 2523, 1195],
    "w": 2536,
    "h": 1230,
    "png": "/home/vlm_train/labeler_feat/output/layout_table_7489/...png",
    "n_table": 1,
    "fallback": false
  }
}
```

레이아웃 labeler converter 결과는 아래 JSONL이다.

```text
/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/_train_data/layout_src_9984/labeler_converter_layout_source_9984.jsonl
```

각 라인은 대략 아래 필드를 가진다.

```json
{
  "id": "...",
  "image_path": "...png",
  "layout_elements": [
    {"bbox": [148, 211, 416, 388], "category": "Page-header", "text": "..."}
  ],
  "layout_json": "[...]",
  "status": "CONVERTED",
  "convert_path": "..."
}
```

### 3.4 table+layout 공통 중간 JSONL 만들기

table source와 layout source를 합쳐 하나의 train/valid/test 공통 중간 JSONL 세트로 만든다. 여기서 말하는 공통 중간 JSONL은 table과 layout을 같은 JSON element 배열 schema로 맞춘 임시 표현이다. 서버의 최종 학습 데이터 디렉터리명에는 더 이상 `unified`라는 표현을 쓰지 않는다.

```bash
cd /NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm
source /NHNHOME/WORKSPACE/0426030039_A/shkim/tsr_test/train/vlm/.venv/bin/activate

python -m data.build_unified_dataset \
  --table-dir ./_train_data/table_src_6902 \
  --layout-jsonl ./_train_data/layout_src_9984/labeler_converter_layout_source_9984.jsonl \
  --layout-image-root ./_train_data/chandra_table_layout_divhtml_16886/images/layout \
  --out-dir ./_train_data/chandra_table_layout_divhtml_16886 \
  --bbox-scale 1024 \
  --copy-images \
  --seed 20260602 \
  --output-format json
```

스크립트 이름은 기존 구현명 때문에 `build_unified_dataset`으로 남아 있지만, 이 문서에서는 이 산출물을 `table+layout 공통 중간 JSONL`이라고 부른다. 이 단계의 산출물은 아직 Chandra div-HTML 최종 target이 아니다.

### 3.5 Chandra div-HTML target으로 변환

아래 단계가 표/레이아웃 데이터를 실제 학습 target인 `div data-bbox/data-label` HTML로 바꾸는 단계다.

```bash
python -m data.build_chandra_dataset \
  --data-dir ./_train_data/chandra_table_layout_divhtml_16886 \
  --manifest ./_train_data/table_bbox_manifest.json \
  --scale 1000 \
  --inplace
```

`--inplace`는 `data/*.jsonl`을 덮어쓰고 백업을 만든다. 새 실험에서는 별도 출력 디렉터리를 두는 편이 안전하다.

```bash
python -m data.build_chandra_dataset \
  --data-dir ./_train_data/chandra_table_layout_divhtml_16886 \
  --manifest ./_train_data/table_bbox_manifest.json \
  --scale 1000 \
  --out-dir ./_train_data/chandra_table_layout_divhtml_16886_regen
```

### 3.6 학습 설정

공통 설정:

- Base model: `/NHNHOME/WORKSPACE/0426030039_A/shkim/models/Qwen3.5-9B`
- Training: SFT + LoRA
- LoRA: `r=128`, `alpha=256`, dropout `0.15`
- LoRA target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- `include_thinking: false`
- `train_ocr_mix_ratio: 0.5`
- learning rate: `1.5e-5`

주요 config:

| config | output | 설명 |
|---|---|---|
| `config/exp_chandra_layout.yaml` | `output/chandra_layout_20260622/student_sft` | layout-only 모델 |
| `config/exp_chandra_table_e18align.yaml` | `output/chandra_table_e18align_20260623/student_sft` | tsr 평가셋용 table-only div-HTML 모델 |
| `config/exp_chandra_all_e18align.yaml` | `output/chandra_all_e18align_20260623/student_sft` | table+layout div-HTML 모델. tsr 200장과 dp bench 500장 평가에 공통 사용 |
| `config/exp_e18_nodiv_20260624.yaml` | `output/e18_nodiv_20260624/student_sft` | div 없는 e18 원본 control |

### 3.7 학습 명령어

layout-only:

```bash
cd /NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm
source /NHNHOME/WORKSPACE/0426030039_A/shkim/tsr_test/train/vlm/.venv/bin/activate

CUDA_VISIBLE_DEVICES=2,3,4 \
NUM_GPUS=3 \
MASTER_PORT=29620 \
WANDB_MODE=offline \
CONFIG=config/exp_chandra_layout.yaml \
PHASE=sft \
bash distill/run_distill.sh
```

tsr 평가셋용 table-only div-HTML:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_GPUS=4 \
MASTER_PORT=29640 \
WANDB_MODE=online \
CONFIG=config/exp_chandra_table_e18align.yaml \
PHASE=sft \
bash distill/run_distill.sh
```

table+layout div-HTML — tsr 200장과 dp bench 500장 평가에 공통 사용한 모델:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
NUM_GPUS=4 \
MASTER_PORT=29650 \
WANDB_MODE=online \
CONFIG=config/exp_chandra_all_e18align.yaml \
PHASE=sft \
bash distill/run_distill.sh
```

e18 nodiv control:

```bash
bash scripts/run_train_e18nodiv.sh
```

### 3.8 평가 명령어

#### tsr 모델 평가 데이터셋 200장 평가

dots.ocr:

```bash
cd /NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm
source /NHNHOME/WORKSPACE/0426030039_A/shkim/tsr_test/train/vlm/.venv/bin/activate

PYTHON_BIN=/NHNHOME/WORKSPACE/0426030039_A/shkim/tsr_test/train/vlm/.venv/bin/python \
bash eval/eval_dots_ocr_6000test.sh
```

table-only div-HTML checkpoint 평가:

```bash
bash scripts/wait_train_and_eval_6000test.sh
```

table+layout div-HTML checkpoint 평가:

```bash
bash scripts/run_eval_tablelayout_all_ckpts_6000test.sh
```

e18 nodiv control 평가:

```bash
bash scripts/wait_train_and_eval_6000test_nodiv.sh
```

평가 결과는 각 run 디렉터리의 `metrics.json`, `predictions.jsonl`, `report.html`을 본다.

#### dp bench 평가 데이터셋 500장 추론

평가 manifest:

```text
eval_data/500_genos_val_set/inference_manifest_with_ocr.jsonl
```

layout-only 추론:

```bash
CONFIG_MARKER=exp_chandra_layout.yaml \
TRAIN_OUTPUT=output/chandra_layout_20260622/student_sft \
OUTPUT_DIR=eval_results/genos500_ocr_layout \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/wait_train_and_infer_genos500_ocr.sh
```

table+layout 추론:

```bash
CONFIG_MARKER=exp_chandra_all_e18align.yaml \
TRAIN_OUTPUT=output/chandra_all_e18align_20260623/student_sft \
OUTPUT_DIR=eval_results/genos500_ocr_tablelayout_epoch3 \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/wait_train_and_infer_genos500_ocr.sh
```

위 명령의 `OUTPUT_DIR` 이름은 새 추론 결과를 저장할 폴더명이므로 필요하면 자유롭게 바꿀 수 있다. 단, 이름을 바꾸면 아래 dp-bench 평가 명령의 `TABLELAYOUT_JSONL`도 같은 경로로 바꿔야 한다.

각 output에는 `predictions_unified.jsonl`, `metrics_unified.json` 계열 파일이 생성된다. 이 단계는 모델 추론만 수행한다. 위 성능표의 `NID`, `TEDS`, `matched/GT`, `extra_pred`는 아래 dp-bench 평가 단계에서 계산한다.

#### dp bench 평가 데이터셋 500장 dp-bench 평가

이 브랜치에는 평가 코드와 dp-bench GT JSON(`genos500_dpbench/genos-500-set/reference_dp_bench.json`)을 함께 올린다. 따라서 브랜치를 b200 서버에 clone하면 GT를 따로 복사하지 않아도 된다. 500장 이미지와 추론 결과 JSONL은 대용량 데이터/결과 산출물이므로 Git에 올리지 않고 b200 서버 경로를 참조한다.

서버에서 새로 받을 때는 예를 들어 아래처럼 본인 작업 공간에 별도 코드
디렉터리로 clone한다(과거 예시였던 `jhshin/tsr_test_genos500_eval` clone은
현재 서버에 없다). 학습 데이터, 모델 output, 추론 결과는 기존 b200 작업
경로(`/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm`)를 그대로
참조한다.

```bash
cd /NHNHOME/WORKSPACE/0426030039_A/<본인 디렉터리>

git clone -b task/318-metric-update \
  https://gitlab.genon.ai/mnc/tsr_test.git \
  tsr_test_eval

cd tsr_test_eval
```

평가에 쓰는 코드:

| 코드 | 역할 |
|---|---|
| `genos500_dpbench/scripts/prepare/convert_qwen_pred_to_dpbench.py` | `predictions_unified.jsonl`의 Chandra div-HTML 출력을 dp-bench prediction JSON으로 변환한다. `<div data-bbox="..." data-label="...">`와 내부 `<table>`을 파싱한다. |
| `genos500_dpbench/upstage-dp-bench/evaluate.py` | task/272 기반 dp-bench 평가기. layout `NID`, table `TEDS/TEDS-S`, bbox IoU table matching을 계산한다. |
| `genos500_dpbench/scripts/run_eval_layout_vs_tablelayout.sh` | `chandra_layout`, `chandra_tablelayout` 변환과 평가를 한 번에 실행하는 wrapper다. optional로 `dots_ocr` baseline도 같이 평가할 수 있다. |

의존성 설치:

```bash
cd /NHNHOME/WORKSPACE/0426030039_A/<본인 디렉터리>/tsr_test_eval
source /NHNHOME/WORKSPACE/0426030039_A/shkim/tsr_test/train/vlm/.venv/bin/activate

python -m pip install -r genos500_dpbench/upstage-dp-bench/requirements.txt
```

평가 입력 준비:

```bash
cd /NHNHOME/WORKSPACE/0426030039_A/<본인 디렉터리>/tsr_test_eval

# GT JSON은 브랜치에 포함되어 있어야 한다.
ls genos500_dpbench/genos-500-set/reference_dp_bench.json

# GT bbox 정규화/매칭에는 모델이 실제 추론한 500장 이미지 크기를 사용한다.
ln -sfn /NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/eval_data/500_genos_val_set/images \
  genos500_dpbench/genos-500-set/qwen_infer_images
```

Chandra 두 모델 평가:

```bash
cd /NHNHOME/WORKSPACE/0426030039_A/<본인 디렉터리>/tsr_test_eval

PYTHON=/NHNHOME/WORKSPACE/0426030039_A/shkim/tsr_test/train/vlm/.venv/bin/python \
TSR_VLM_ROOT=/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm \
REF=genos500_dpbench/genos-500-set/reference_dp_bench.json \
IMAGES_DIR=/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/eval_data/500_genos_val_set/images \
LAYOUT_JSONL=/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/eval_results/genos500_ocr_layout/predictions_unified.jsonl \
TABLELAYOUT_JSONL=/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/eval_results/genos500_ocr_tablelayout_epoch3/predictions_unified.jsonl \
RUN_DOTS=false \
bash genos500_dpbench/scripts/run_eval_layout_vs_tablelayout.sh
```

위 성능표를 그대로 재현할 때의 table+layout 추론 결과는 `genos500_ocr_tablelayout_epoch3`이다. 다른 checkpoint/output을 평가하려면 `TABLELAYOUT_JSONL`만 해당 `predictions_unified.jsonl` 경로로 바꾼다.

출력:

```text
genos500_dpbench/dp_out/genos500_chandra_ocr_20260623/layout/pred_dp_bench_full.json
genos500_dpbench/dp_out/genos500_chandra_ocr_20260623/layout/eval_result_gitlab.txt
genos500_dpbench/dp_out/genos500_chandra_ocr_20260623/tablelayout/pred_dp_bench_full.json
genos500_dpbench/dp_out/genos500_chandra_ocr_20260623/tablelayout/eval_result_gitlab.txt
```

`dots_ocr` baseline까지 같은 표로 재현하려면 dots.ocr의 dp-bench prediction JSON을 준비한 뒤 `RUN_DOTS=true`와 `DOTS_PRED`를 지정한다. dots.ocr prediction은 이미 dp-bench JSON 형태이므로 Chandra처럼 `predictions_unified.jsonl` 변환을 거치지 않는다.

```bash
RUN_DOTS=true \
DOTS_PRED=genos500_dpbench/analysis_results/dots_pred_dpv2_original.json \
bash genos500_dpbench/scripts/run_eval_layout_vs_tablelayout.sh
```

평가 설정:

| 모델 | pred 좌표계 | pred grid | NID 설정 | TEDS 매칭 |
|---|---|---:|---|---|
| `dots_ocr` | `fraction_bl` | 1024 | table/figure/chart 제외, table text conversion off | bbox IoU≥0.5 |
| `chandra_layout` | `grid` | 1000 | table/figure/chart 제외, table text conversion off | bbox IoU≥0.5 |
| `chandra_tablelayout` | `grid` | 1000 | table/figure/chart 제외, table text conversion off | bbox IoU≥0.5 |

좌표계(`fraction_bl`/`grid`/`image`)가 무엇이고 왜 통일 변환이 필요한지는 [genos500_dpbench/docs/coord_spaces.md](genos500_dpbench/docs/coord_spaces.md) 참고.

재현되는 성능:

| 모델 | NID | TEDS zero(b) | TEDS-S zero(b) | TEDS skip(a) | TEDS-S skip(a) | matched/GT | extra_pred |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dots_ocr` | 0.8836 | 0.7923 | 0.8020 | 0.8219 | 0.8320 | 294/305 | 16 |
| `chandra_layout` | 0.8903 | 0.6949 | 0.7175 | 0.7679 | 0.7929 | 276/305 | 34 |
| `chandra_tablelayout` | 0.8905 | 0.6999 | 0.7225 | 0.7936 | 0.8192 | 269/305 | 30 |
| `mineru_native` | 0.9353 | 0.7259 | 0.7484 | 0.7796 | 0.8037 | 284/305 | 32 |
| `mineru_hybrid` | 0.9026 | 0.7100 | 0.7300 | 0.7874 | 0.8096 | 275/305 | 19 |
| `mineru_lora` | 0.9305 | 0.7251 | 0.7482 | 0.7760 | 0.8007 | 285/305 | 20 |
| `paddle` | 0.9019 | 0.6897 | 0.7129 | 0.7763 | 0.8024 | 271/305 | 27 |

### 3.9 신규 평가 metric (이슈 doc_parser#318)

기존 NID/TEDS에 더해 OmniDocBench v1.7 rule-based(Text/ReadingOrder Edit, CDM,
TEDS)와 pdf-parse-bench LLM-Judge(수식·표 0~10점)를 추가했다. **upstream 평가기는
수정 없이 원본 그대로 쓰고, 이 브랜치의 어댑터가 우리 GT/pred를 그 포맷으로
변환하는 구조**다 (검증된 매칭 알고리즘을 왜곡하지 않기 위한 의도적 선택 —
`evaluate.py` 이식은 하지 않는다).

아래 순서대로 실행하면 1.2·1.3의 신규 metric 수치가 그대로 재현된다.
3.8의 기존 평가와는 **별도 실행**이다(3.8을 돌려도 신규 metric은 안 나온다).

#### 사전 준비 (1회)

이 브랜치 clone(`tsr_test_eval`)과 upstream 평가기 2종을 나란히 두고, upstream
venv를 각각 만든다(파이썬 버전이 서로 달라 venv 분리 필수). 3.8을 이미
진행했다면 그 `tsr_test_eval`을 그대로 쓰면 되고, clone 단계는 건너뛴다:

```bash
W=/NHNHOME/WORKSPACE/0426030039_A/<본인 디렉터리>

# 이 브랜치 (3.8에서 이미 clone했다면 생략)
git clone -b task/318-metric-update \
  https://gitlab.genon.ai/mnc/tsr_test.git \
  $W/tsr_test_eval

git clone --depth 1 https://github.com/opendatalab/OmniDocBench.git $W/upstream_OmniDocBench
cd $W/upstream_OmniDocBench && uv venv --python 3.11 .venv && uv pip install -p .venv/bin/python -e .

git clone --depth 1 https://github.com/phorn1/pdf-parse-bench.git $W/upstream_pdf_parse_bench
cd $W/upstream_pdf_parse_bench && uv venv --python 3.12 .venv && uv pip install -p .venv/bin/python -e .

cd $W/tsr_test_eval
```

- **CDM까지 돌리려면**: [CDM_SETUP_b200.md](genos500_dpbench/omnidocbench/CDM_SETUP_b200.md)의
  1~4단계(TeX Live 유저 설치·magick shim·한글 렌더 테스트)를 먼저 완료한다.
  CDM 없이 Edit/TEDS/읽기순서만 볼 거면 생략 가능(`WITH_CDM=1`만 빼면 됨).
- **LLM-Judge용 인증키(`<AUTH_KEY>`)**: GenOS 서빙 776
  (Qwen3.5-397B-FP8-Instruct, 사내 자체 서빙이라 과금 없음) 상세 페이지
  <https://genos.genon.ai/serving/model/detail/776/?modelType=VLM> 의
  "인증 키" 탭에서 **메모가 "전처리기 평가 metric LLM-Judge (doc_parser#318)"인
  ID 1274 키**의 값을 아래 `<AUTH_KEY>` 자리에 넣는다 (키가 안 보이면 같은
  탭에서 본인 키를 새로 발급해도 됨).

#### ① OmniDocBench 4종(+CDM) — 3모델 일괄

**CDM을 포함한다면 먼저 아래 확인부터.** `TEX_ROOT`는 작업 디렉터리(`$W`)가
아니라 [CDM_SETUP_b200.md](genos500_dpbench/omnidocbench/CDM_SETUP_b200.md)에서
TeX Live·magick을 **설치한 디렉터리**다. PATH에 pdflatex가 없으면 CDM이
에러로 죽지 않고 **조용히 전부 0으로 나오므로**, `which` 두 개가 모두
`$TEX_ROOT` 경로로 출력되는 것을 확인하기 전에는 진행하지 말 것:

```bash
TEX_ROOT=<TeX Live·magick을 설치한 디렉터리>
export PATH=$TEX_ROOT/texlive/2025/bin/x86_64-linux:$TEX_ROOT/bin:$PATH
export CDM_CJK_FONT=mj

which pdflatex magick   # 둘 다 $TEX_ROOT 밑 경로가 나와야 함
```

CDM 없이 Edit/TEDS/읽기순서만 볼 거면 위 블록은 건너뛰고, 아래에서
`WITH_CDM=1 \` 줄만 지우면 된다.

본 실행 (같은 셸에서 이어서):

```bash
W=/NHNHOME/WORKSPACE/0426030039_A/<본인 디렉터리>
cd $W/tsr_test_eval

WITH_CDM=1 \
OMNIDOCBENCH_ROOT=$W/upstream_OmniDocBench \
GT_JSON=genos500_dpbench/genos-500-set/reference_dp_bench.json \
IMAGES_DIR=/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/eval_data/500_genos_val_set/images \
DOTS_PRED=genos500_dpbench/analysis_results/dots_pred_dpv2_original.json \
CHANDRA_LAYOUT_JSONL=/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/eval_results/genos500_ocr_layout/predictions_unified.jsonl \
CHANDRA_TABLELAYOUT_JSONL=/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/eval_results/genos500_ocr_tablelayout_epoch3/predictions_unified.jsonl \
bash genos500_dpbench/omnidocbench/run_omnidocbench_genos500.sh
```

결과는 `$W/upstream_OmniDocBench/result/pred_md_<모델>_quick_match_metric_result.json`
(`display_formula`에 CDM, `table`에 TEDS, `text_block`/`reading_order`에 Edit).
per-element 매칭 쌍 덤프(`..._table_result.json` 등)도 같이 생긴다.

#### ② LLM-Judge — dp bench 평가 데이터셋 500장

①이 만든 markdown 변환본(`omnidocbench_out/pred_md_<모델>/`)을 재사용하므로
**① 다음에** 실행한다. `<AUTH_KEY>`는 [서빙 776 상세 페이지](https://genos.genon.ai/serving/model/detail/776/?modelType=VLM)
"인증 키" 탭의 **ID 1274 키**(메모: "전처리기 평가 metric LLM-Judge
(doc_parser#318)") 값이다 (③·④도 동일):

```bash
PYPPB=$W/upstream_pdf_parse_bench/.venv/bin/python

# GT 변환 (1회)
$PYPPB genos500_dpbench/pdfparsebench/convert_gt_to_ppb.py \
  --gt genos500_dpbench/genos-500-set/reference_dp_bench.json \
  --output_dir genos500_dpbench/ppb_out/ground_truth

# 3모델 매칭+채점 (모델당 ~10분, 중단돼도 재실행하면 이어서 함)
for TAG in dots_ocr chandra_layout chandra_tablelayout; do
  $PYPPB genos500_dpbench/pdfparsebench/run_ppb_genos500.py \
    --gt_dir genos500_dpbench/ppb_out/ground_truth \
    --pred_md_dir genos500_dpbench/omnidocbench_out/pred_md_$TAG \
    --out_dir genos500_dpbench/ppb_out/$TAG \
    --base_url https://genos.genon.ai/api/gateway/rep/serving/776/v1 \
    --api_key <AUTH_KEY> --model model --max_workers 4 --step all
done
```

말미에 `[formula]`/`[table]` 요약(매칭 수·평균)이 출력되고, 항목별 결과는
`ppb_out/<모델>/<문서>/{formulas,tables}.json`에 남는다.

주의: LLM-Judge는 매칭·채점을 모두 LLM이 하므로 **재실행 시 수치가 다소
달라진다** (재현 실측: 표 평균 ±0.3~0.4, 수식은 표본 30개라 ±1 이상 가능.
매칭 수도 몇 건씩 변동). 1.2 성능표의 값과는 경향(모델 서열, TEDS 대비
낮은 절대값)이 일치하면 정상 재현으로 본다. Edit/TEDS/CDM은 결정적이라
자리수까지 일치해야 한다.

#### ③ LLM-Judge — tsr 모델 평가 데이터셋 200장

3.8의 각 그룹 대표 run(final)을 자동 선택해 채점한다 (run당 ~2분).
새 셸에서 시작한다면 `W`/`PYPPB`부터 다시 정의할 것 — `$PYPPB`가 비어 있으면
`Permission denied`(py 파일 직접 실행 시도)가 난다:

```bash
W=/NHNHOME/WORKSPACE/0426030039_A/<본인 디렉터리>
cd $W/tsr_test_eval
PYPPB=$W/upstream_pdf_parse_bench/.venv/bin/python

EVAL=/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/eval_results

for GROUP in dots_ocr_6000test table_6000test tablelayout_6000test e18_nodiv_6000test; do
  P=$(ls -t $EVAL/$GROUP/*/predictions.jsonl 2>/dev/null | grep final | head -1)
  [ -z "$P" ] && P=$(ls -t $EVAL/$GROUP/*/predictions.jsonl 2>/dev/null | head -1)
  echo "===== $GROUP ($P)"
  $PYPPB genos500_dpbench/pdfparsebench/run_ppb_tsr.py \
    --predictions "$P" \
    --out_dir genos500_dpbench/ppb_out/tsr200_$GROUP \
    --base_url https://genos.genon.ai/api/gateway/rep/serving/776/v1 \
    --api_key <AUTH_KEY> --model model --max_workers 4
done
```

run별로 전체/complexity별 평균과 TEDS 구간별 LLM 평균이 출력되고, 항목별
비교는 `ppb_out/tsr200_<그룹>/llm_vs_teds.csv`에 남는다.

#### ④ (선택) 표 매칭 방식 비교 · 수식 문법 강건성 실험

```bash
# dp-bench 평가 의존성 venv (1회)
uv venv --python 3.11 $W/venv_dpbench
uv pip install -p $W/venv_dpbench/bin/python -r genos500_dpbench/upstage-dp-bench/requirements.txt

# 표 매칭 비교 (bbox vs 텍스트 vs LLM) — ①·② 완료 후
PY_EVAL=$W/venv_dpbench/bin/python \
OMNIDOCBENCH_ROOT=$W/upstream_OmniDocBench \
GT_JSON=$W/tsr_test_eval/genos500_dpbench/genos-500-set/reference_dp_bench.json \
IMAGES_DIR=/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/eval_data/500_genos_val_set/images \
DOTS_PRED=$W/tsr_test_eval/genos500_dpbench/analysis_results/dots_pred_dpv2_original.json \
DOTS_SPACE=fraction_bl \
CHANDRA_LAYOUT_JSONL=/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/eval_results/genos500_ocr_layout/predictions_unified.jsonl \
CHANDRA_TABLELAYOUT_JSONL=/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/eval_results/genos500_ocr_tablelayout_epoch3/predictions_unified.jsonl \
bash genos500_dpbench/analysis/run_table_match_compare_b200.sh

# 수식 문법 강건성 실험 (LLM 축; CDM 축은 산출된 cdm_gt.json을 ①의 평가기로)
$PYPPB genos500_dpbench/analysis/formula_variant_experiment.py \
  --gt genos500_dpbench/genos-500-set/reference_dp_bench.json \
  --out_dir genos500_dpbench/formula_variants_out \
  --base_url https://genos.genon.ai/api/gateway/rep/serving/776/v1 \
  --api_key <AUTH_KEY> --model model
```

측정 결과는 1.2(500장)·1.3(200장) 성능표에 통합했다. tsr 200장 complexity별
LLM 평균:

| run | simple(30) | medium(50) | complex(40) | complex_col(30) | complex_mix(30) | complex_row(20) |
|---|---:|---:|---:|---:|---:|---:|
| `dots_ocr` | 8.37 | 5.42 | 2.95 | 4.83 | 3.43 | 2.90 |
| `table (final)` | 9.47 | 8.48 | 5.58 | 6.27 | 5.53 | 4.80 |
| `tablelayout (final)` | 9.47 | 8.04 | 5.75 | 6.73 | 5.33 | 4.15 |
| `e18_nodiv (final)` | 9.40 | 8.52 | 5.80 | 6.73 | 5.77 | 5.00 |

metric 신뢰성 분석 요약: 수식은 렌더 동일 문법 변형 108쌍에서 CDM 1.0·LLM
108/108 만점, Edit는 정규화 후에도 3~5% 잔여 벌점 → **수식 주 지표는 CDM
권고**. 표는 TEDS의 상향 편향(1.2 해석 참조) 때문에 **TEDS 유지 + LLM-Judge
병행 보고 권고**.

**결과·분석 문서** (`genos500_dpbench/analysis_results/`):

- [EVAL_RESULTS.md](genos500_dpbench/analysis_results/EVAL_RESULTS.md) — 전체 수치 종합표 (reeval 참고치·매칭 3방식 비교·TEDS 구간별 정렬 포함)
- [PHASE3_REPORT.md](genos500_dpbench/analysis_results/PHASE3_REPORT.md) — 신뢰성 비교 분석·두 데이터셋 역할·도입 권고안

## 4. 결과 해석 시 주의사항

- `table_6000test`는 table-only div-HTML 모델 평가 결과다.
- 성능표의 `dots_ocr` 산출물 원본은
  `genos500_dpbench/analysis_results/dots_pred_dpv2_original.json`으로 브랜치에
  보존했다(NID·TEDS 4수치 정확 재현 확인). jhyeo의 `genos500_dots_reeval`은
  이보다 성능이 크게 좋은 **별개 추론분**(NID 0.9544)이므로 혼용하지 말 것.
- TEDS는 검출 실패 표 제외·다단 헤더 부분점수 때문에 후한 방향의 편향이
  있다 — 표 품질 판단 시 LLM-Judge 수치(1.2·1.3)를 병행 참조. 수식 비교에는
  Edit 대신 CDM을 쓴다.
- `tablelayout_6000test`는 table+layout 통합 div-HTML 모델 평가 결과다.
- `genos500_ocr_layout`은 layout-only 모델의 dp bench 평가 데이터셋 500장 추론 결과다.
- `genos500_ocr_tablelayout_epoch3`은 위 dp bench 성능표의 `chandra_tablelayout` 수치를 재현하는 table+layout 모델 추론 결과다. 다른 checkpoint/output을 평가할 때는 README의 `TABLELAYOUT_JSONL`만 바꿔 실행한다.
- `e18_6910 ckpt-500`은 비교 기준으로 사용했지만 이 브랜치의 `output/`에는 포함하지 않는다.
- Git에는 `_train_data/`, `eval_data/`, `output/`, `eval_results/`, `wandb/`, `logs/`를 올리지 않는다.

새 사람이 이 브랜치를 받으면 먼저 b200 서버의 데이터/모델 경로가 그대로 존재하는지 확인하고, `train/vlm`에서 위 명령을 순서대로 실행하면 된다.
