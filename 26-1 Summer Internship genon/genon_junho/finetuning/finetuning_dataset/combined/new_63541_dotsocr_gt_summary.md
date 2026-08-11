# new_63541: dots-ocr 추출 & GT 비교 요약

## 배경
`new_63541` 데이터셋(gt_html_dataset.jsonl, 20,932장 = train+valid+test 합)의 GT가 잘못됐을
가능성이 의심되어, 동일 이미지를 dots-ocr(dots.mocr, vLLM)로 다시 추출해 GT와 대조했다.

## 한 일
1. **dots-ocr 재추출**: GPU 0/2/3에 dots.mocr vLLM 서버 3개를 띄워 20,932장 병렬 추출 (약 55분 소요, 에러 0건).
2. **GT bbox 역변환**: gt_html의 bbox가 실제 이미지 크기와 무관하게 0~1000으로 정규화되어 있음을 확인.
   각 이미지 실제 픽셀 크기로 역변환(`x_px = x_norm/1000*width`)해서 dots-ocr(픽셀 스케일)과 비교 가능하게 만듦.
3. **텍스트/카테고리 diff 분석**: bbox가 아니라 **텍스트 내용**(표 안/밖 모두 태그 제거 후 순수 텍스트)을 기준으로
   GT와 dots-ocr 영역을 정렬(매칭)한 뒤, (a) 문서 전체 텍스트 유사도, (b) 매칭된 영역들의 카테고리 일치 여부를 계산.

## 산출물 (jsonl, `combined/` 폴더)
- `new_63541_label_from_dotsocr.jsonl` — dots-ocr 원본 추출 결과 (20,932장, `image_path`/`gt_html`/`full_response`/`dotsocr_html`)
- `new_63541_gt_html_pixelscale.jsonl` — GT bbox를 픽셀 스케일로 역변환한 버전 (`image_width`/`image_height` 포함)
- `new_63541_gt_vs_dotsocr_diff.jsonl` — (초기 버전) bbox IoU 매칭 기반 diff 리포트. n_gt/n_pred 개수 차이에 가중치가 커서 "문장 단위 vs 문단 단위" 같은 granularity 차이도 의심도가 높게 나오는 한계가 있었음.
- `new_63541_text_cat_diff.jsonl` — **최종 버전**. 텍스트 유사도로 영역을 정렬해 카테고리 일치를 보는 방식. 문서별 필드:
  `doc_text_sim`(전체 텍스트 유사도), `cat_matched`/`cat_agree`/`cat_disagree`/`cat_match_rate`, `n_gt_only_text`/`n_pred_only_text`, `disagreements`(카테고리 불일치 상세).

## 핵심 발견

### 1. bbox 스케일 불일치 (GT vs dots-ocr 원본, vs 파인튜닝 학습 컨벤션)
- GT(`gt_html`)의 bbox는 이미지 실제 크기와 무관하게 항상 0~1000 정규화 그리드 (`bbox_scale=1000`, train/valid/test/train_refined 전체 100% 일관).
- dots-ocr 원본 출력은 정규화 없이 이미지 네이티브 픽셀 스케일에 가깝게 나옴(모델 pretrained prior).
- 파인튜닝 프롬프트(`vlm/utils/prompt_templates.py`)는 강제로 "0~1000 정규화"를 지시하는데, `_render_prompt_body()`가 실제로는 `bbox_scale` 인자를 무시하고 항상 고정 텍스트를 씀 (지금은 데이터가 전부 1000이라 문제 없지만 취약한 구조).
- 학습 이미지 300장 샘플링 결과 약 83%가 가로세로비 0.69~0.79(A4 스캔)에 쏠려있어, 이 비율 밖의 새 문서에서 모델이 정규화를 제대로 일반화하지 못했을 가능성.
- eval 인프라(`upstage-dp-bench/src/table_evaluation.py`)는 이미 `"image"`(픽셀 스케일) coord space를 지원하므로, **GT/학습 타깃을 픽셀 스케일로 바꾸는 마이그레이션은 코드 수정 없이 가능**해 보임 (이미 만든 `new_63541_gt_html_pixelscale.jsonl`이 그 변환 예시).

### 2. 텍스트 내용 비교 (`new_63541_text_cat_diff.jsonl` 기준, 20,932건)
- 전체 텍스트 유사도 < 0.9: 2,158건 / < 0.7: 764건 / < 0.5: 480건 / < 0.3: ~310건
- GT 또는 dots-ocr 어느 한쪽이 영역을 아예 0개 잡은 문서: 43건 (GT가 0개: 6건, dots-ocr가 0개: 37건)

### 3. 카테고리 매칭 (텍스트로 정렬된 쌍 기준)
- 평균 카테고리 일치율(cat_match_rate): 0.831
- 카테고리 불일치가 있는 문서: 10,911건
- **가장 흔한 혼동 패턴**: `list-item ↔ text` (압도적 1위, 2만건 이상), 그 다음 `footnote→text`, `title→section-header`, `page-header↔section-header`, `caption→text`, `table→text` 등.
  → 텍스트 내용 자체는 맞는데 라벨링 기준(GT vs dots-ocr) 차이인 경우가 많아, "GT가 틀렸다"기보다 **라벨링 가이드라인 불일치**로 보이는 케이스가 다수.

### 4. 파서 버그 (발견 후 수정 완료)
- 초기 diff 스크립트가 dots-ocr 응답의 text 필드 안에 ` ```python ... ``` ` 같은 코드블록이 포함된 경우, 이를 마크다운 펜스로 오인해 전체 JSON 파싱이 깨지는 버그가 있었음 (9건 영향, 예: `CZ CELLxGENE Discover (2025) NAR_page_013.png`가 `n_pred=0`으로 잘못 집계됨). 원본 텍스트를 보존하는 방식으로 수정 후 재생성 완료.

## 다음에 볼 만한 것
- `new_63541_text_cat_diff.jsonl`에서 `doc_text_sim` 오름차순 정렬 → 실제로 내용이 다른 문서부터 확인
- `disagreements` 필드로 카테고리 혼동 패턴별 빈도 재확인 (라벨링 가이드 정합성 점검용)
- 학습 시 bbox를 픽셀 스케일로 바꿔서 재학습해보는 A/B 검증 (근거는 위 "1. bbox 스케일 불일치" 참고)
