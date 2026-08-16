# 통합 학습 파이프라인 (Table + Layout) — 처음 보는 사람도 이해하는 전체 가이드

이 문서는 **테이블 데이터**와 **레이아웃 데이터**를 한 모델에 함께 학습시켜,
두 경우 모두 추출 결과가 **똑같은 JSON 형태**로 나오게 만든 작업 전체를 정리한다.

처음 보는 팀원도 한 번에 이해할 수 있도록 다음 순서로 설명한다.

1. 기본 개념 (VLM / TSR / LoRA가 뭔지)
2. 통합 전, 원본 데이터가 어떻게 생겼는지 (테이블 vs 레이아웃)
3. 왜 통합이 필요한지 + 통합 출력 스키마
4. 전체 파이프라인 한눈에 보기
5. 학습 코드 안에서 데이터가 어떻게 흘러가는지 (단계별 상세)
6. 평가(성능 측정) — 새로 만든 지표와 실행법
7. 변경/추가된 파일
8. 실행 명령어 모음
9. 주의사항 / 알려진 제약
10. 용어집

---

## 1. 기본 개념 먼저

### 1-1. VLM (Vision-Language Model, 이미지를 이해하는 언어모델)
- 일반 LLM(GPT 같은 것)은 텍스트만 입력받지만, **VLM은 이미지 + 텍스트를 같이 입력**받아
  텍스트를 생성한다.
- 우리가 쓰는 모델은 **Qwen3-VL** 계열(`Qwen3.5-9B`). 이미지를 "패치(작은 조각)"로 잘라
  숫자 벡터로 만든 뒤, 텍스트 토큰과 함께 트랜스포머에 넣는다.
- 입력: `이미지 + 지시문(프롬프트)` → 출력: `우리가 원하는 형식의 텍스트`.

### 1-2. TSR (Table Structure Recognition, 표 구조 인식)
- "표 이미지"를 보고 **표의 구조(행/열/병합셀)와 내용을 HTML로 복원**하는 작업.
- 예: 표 사진 → `<table><tr><td>이름</td><td>점수</td></tr>...</table>`
- 병합셀은 `colspan`(가로 병합), `rowspan`(세로 병합)으로 표현한다.

### 1-3. 레이아웃 인식 (Document Layout Analysis)
- "문서 페이지 한 장"을 보고 **어떤 영역이 제목인지/본문인지/표인지/그림인지**를
  네모(bbox)와 종류(category)로 찾아내는 작업.
- 예: 페이지 사진 → `제목 영역(여기), 본문 영역(여기), 표 영역(여기)...`

### 1-4. LoRA / QLoRA (효율적 파인튜닝 기법) — 가장 중요
모델 전체(수십억 개 파라미터)를 다시 학습하면 GPU 메모리·시간이 엄청나게 든다.
**LoRA는 원본 모델은 얼리고(freeze), 작은 보조 행렬만 학습**하는 방법이다.

- 비유: 큰 사전(원본 모델)은 그대로 두고, 그 옆에 **얇은 포스트잇 묶음(LoRA 어댑터)**만
  새로 적어 붙인다. 추론할 때 사전 + 포스트잇을 같이 본다.
- 수식적으로: 어떤 가중치 `W`(고정)에 대해 `W + B·A` 를 쓴다. 여기서 `A`, `B`가
  작은 행렬(rank `r`만큼)이고, **학습되는 건 `A`, `B`뿐**.
- `lora_r`(rank): 보조 행렬 크기. 클수록 표현력↑·용량↑. 우리는 `128`.
- `lora_alpha`: 보조 행렬의 영향 배율(스케일). 우리는 `256`.
- `target_modules`: 어느 레이어에 어댑터를 붙일지. 우리는 attention(q/k/v/o) +
  MLP(gate/up/down) 전체.
- **QLoRA** = 원본 모델을 **4-bit로 양자화(압축)** 해서 메모리를 더 줄인 뒤 LoRA를 올린 것.
  학습 산출물은 보통 **수백 MB짜리 어댑터** 하나다(원본 모델 9B는 그대로 둠).

> 핵심: 우리가 학습하는 건 "통합 출력을 내도록 시키는 작은 어댑터"이고, 평가/추론 시에는
> `원본 9B 모델 + 이 어댑터`를 함께 로드한다.

---

## 2. 통합 전 — 원본 데이터는 어떻게 생겼나

두 데이터는 **완전히 다른 형태**다. 그래서 그대로는 한 모델에 못 넣는다.

### 2-1. 테이블 데이터 (약 7,489개, 전부 OCR 있음)
한 줄(JSONL 1 line)이 표 이미지 1장에 대응한다. 핵심 필드:

```json
{
  "image_path": "images/abc123.jpg",
  "gt_html": "<table><thead><tr><td>항목</td><td colspan=\"2\">값</td></tr></thead>...</table>",
  "ocr_info": [
    {"text": "항목", "bbox": [12, 8, 60, 30]},
    {"text": "값",   "bbox": [120, 8, 180, 30]}
  ],
  "bbox_scale": 1024
}
```

- `image_path`: 표 이미지 (이미지 전체가 표 하나).
- `gt_html`: **정답 = 표 HTML 문자열**.
- `ocr_info`: OCR(글자 인식기)이 미리 뽑아 둔 `텍스트 + 위치(bbox)` 목록.
  표는 글자가 많아서 OCR을 같이 넣어주면 셀 내용을 더 정확히 맞춘다.
- `bbox_scale`: 좌표 정규화 기준(1024). bbox는 0~1024 범위 정수.

→ 입력 = **이미지 + OCR**, 정답 = **HTML**.

### 2-2. 레이아웃 데이터 (현재 약 80개, OCR 없음)
한 줄이 문서 페이지 1장에 대응한다. dots.mocr 도구가 뽑은 element 배열이 들어 있다:

```json
{
  "image_path": "page_0007.png",
  "layout_elements": [
    {"bbox": [60, 40, 540, 90],  "category": "Title",   "text": "2024 재무 보고"},
    {"bbox": [60, 120, 540, 300],"category": "Text",    "text": "본문 문단 ..."},
    {"bbox": [60, 320, 540, 600],"category": "Table",   "text": "<table>...</table>"}
  ]
}
```

- `image_path`: 페이지 이미지 한 장(표/그림/글이 섞여 있음).
- `layout_elements`: **정답 = 여러 영역의 배열**. 각 영역은 `bbox + category + text`.
  - bbox는 **픽셀 좌표**(정규화 안 됨)라는 점이 테이블과 다르다.
  - 표 영역이면 `text`에 HTML이 들어 있다.
- OCR 정보는 없다(dots.mocr가 이미지만으로 추론).

→ 입력 = **이미지만**, 정답 = **element 배열**.

### 2-3. 한눈에 비교

| 항목 | 테이블 데이터 | 레이아웃 데이터 |
|------|---------------|------------------|
| 이미지 | 표 1개 crop | 문서 페이지 전체 |
| 입력 보조정보 | OCR 있음 | 없음 |
| 정답 형태 | HTML 문자열 | element 배열(bbox/category/text) |
| bbox 좌표 | 0~1024 정규화 | 픽셀 좌표 |
| 개수 | ~7,489 | ~80 (스모크용) |

---

## 3. 왜 통합하나 + 통합 출력 스키마

### 3-1. 문제
기존 TSR 모델은 표만 받아서 **HTML만** 뱉었다. 이제 레이아웃 데이터도 같이 넣으려면,
**모델이 두 입력에 대해 서로 다른 형태로 답하면 학습/평가가 꼬인다.**
→ 그래서 **두 경우 모두 같은 JSON 형태로 답하도록 통일**한다.

### 3-2. 통합 출력 스키마 (모델이 내야 하는 정답 형태)

```json
[
  {"bbox": [x0, y0, x1, y1], "category": "<라벨>", "text": "<내용>"},
  ...
]
```

- `bbox`: 0~`bbox_scale`(=1024)로 정규화된 정수 좌표.
- `category`: 영역 종류. 통합 라벨은 11종으로 통일한다: `Title`, `Text`, `Section-header`, `List-item`, `Table`, `Caption`, `Footnote`, `Formula`, `Picture`, `Page-header`, `Page-footer`.
- `text`: 블록 텍스트. **`category=="Table"`이면 `text`가 표 HTML**(`<table>…</table>`).

### 3-3. 두 데이터가 이 스키마로 어떻게 표현되나

| 데이터 | 통합 출력 |
|--------|-----------|
| 테이블 | 원소 1개짜리 배열: `[{"bbox":[0,0,1024,1024], "category":"Table", "text":"<table>…</table>"}]` |
| 레이아웃 | 여러 원소 배열 (Table 원소의 text는 HTML) |

> **입력 정책(확정)**: 레이아웃 데이터는 OCR이 없으므로 **이미지만** 넣고,
> 테이블 데이터는 OCR이 있으므로 **이미지 + OCR**을 넣는다. OCR 유무가 두 태스크의
> 자연스러운 입력 신호가 된다.

---

## 4. 전체 파이프라인 한눈에 보기

```
[테이블 JSONL]                 [레이아웃 JSONL]
 image_path, gt_html(HTML),     image_path,
 ocr_info, bbox_scale           layout_elements(bbox/category/text)
        │                                │
        └────────────────┬───────────────┘
                         ▼
        (1) data/build_unified_dataset.py   ── 통합 데이터셋 빌드
                         ▼
   _train_data/unified_*/data/{train,valid,test}.jsonl
     각 줄: { image_path, task_type, gt_html(=JSON 문자열),
              prompt_style, ocr_info?, bbox_scale }
                         ▼
        (2) scripts/train_unified.sh        ── 학습 실행
          └ distill/run_distill.sh (PHASE=sft)
            └ distill/student_sft.py
               ├ TSRDataset           (task_type 보존 / 레이아웃 가드)
               ├ task 1:1 WeightedRandomSampler
               └ MultimodalCollator → build_chat_messages
                  · system = UNIFIED_SYSTEM_PROMPT
                  · user   = unified 프롬프트 (+ 테이블이면 OCR)
                  · assistant 정답 = gt_html(JSON 문자열)
                         ▼
                  LoRA 어댑터 학습 (체크포인트 저장)
                         ▼
        (3) scripts/eval_unified.sh         ── 성능 측정
          └ eval/evaluate_unified.py
            └ eval/unified_metrics.py
               · 테이블: TEDS / Span F1
               · 레이아웃: element F1 / IoU / category acc / text 유사도
```

---

## 5. 학습 코드 안에서 데이터가 어떻게 흘러가나 (단계별 상세)

### 5-0. 통합 레코드 한 줄 예시 (빌드 결과)
```json
// 테이블 샘플
{"image_path":"images/table/abc.jpg","task_type":"table",
 "gt_html":"[{\"bbox\":[0,0,1024,1024],\"category\":\"Table\",\"text\":\"<table>...</table>\"}]",
 "prompt_style":"unified_table_with_ocr","ocr_info":[...],"bbox_scale":1024}

// 레이아웃 샘플
{"image_path":"page_0007.png","task_type":"layout",
 "gt_html":"[{\"bbox\":[60,40,540,90],\"category\":\"Title\",\"text\":\"...\"}, ...]",
 "prompt_style":"unified_layout","bbox_scale":1024}
```
> 핵심 트릭: 기존 학습 코드가 `gt_html`을 "정답 텍스트"로 쓰기 때문에, **통합 JSON
> 문자열을 그대로 `gt_html`에 넣어** 기존 파이프라인을 거의 그대로 재사용한다.

### 5-1. 데이터 빌드 (`data/build_unified_dataset.py`)
- 테이블: `gt_html(HTML)` → `[{category:"Table", text:HTML, bbox:[0,0,scale,scale]}]` 로 감싸고
  JSON 문자열로 만들어 `gt_html`에 저장. OCR 유무로 `prompt_style` 결정
  (`unified_table_with_ocr` / `unified_table_without_ocr`).
- 레이아웃: `layout_elements`의 **픽셀 bbox를 이미지 크기로 0~scale 정규화**한 뒤
  JSON 배열 문자열로 저장. `prompt_style = unified_layout`.
- 레이아웃은 원래 train/valid/test 구분이 없어 `--layout-valid-ratio/--layout-test-ratio`로 분할.
  테이블은 기존 split을 그대로 사용. 마지막에 둘을 **합쳐서 섞어** 저장.

### 5-2. 데이터셋 로딩 (`train/train_qlora.py`의 `TSRDataset`)
- JSONL을 읽어 레코드 리스트로 보관. `task_type`을 그대로 **보존**한다.
- **레이아웃 가드 1**: `annotate_empty_cells`(표 빈 셀 채우기)는 표 전용이라
  `task_type == "layout"`이면 건너뛴다(코드: `record.get("task_type") != "layout"`).
- **레이아웃 가드 2**: OCR 믹싱(`_apply_ocr_mix`)도 레이아웃은 제외 → 레이아웃의
  `prompt_style(unified_layout)`이 망가지지 않는다.
- **task 카운트/가중치**:
  - `get_task_counts()` → `{"table": N, "layout": M}`
  - `get_task_sampling_weights(ratios)` → 각 샘플 가중치 = `목표비율 / 해당 task 개수`.
    기본은 1:1. 즉 테이블 7,489개는 각자 작은 가중치, 레이아웃 80개는 각자 큰 가중치를
    받아 **뽑힐 기대값이 50:50**이 된다.

### 5-3. 1:1 균형 샘플링 (`distill/student_sft.py`)
- config에서 `task_mix_sampling: true`면 `WeightedRandomSampler`를 만들어
  매 스텝 테이블:레이아웃 = 1:1 로 뽑는다.
- `max_train_samples`로 데이터가 `Subset`이 되어도 동작하도록 처리.
- 레이아웃이 80개뿐이라 epoch마다 여러 번 재노출되지만, **두 태스크의 학습 신호 균형**을
  우선한다. 비율은 `task_sampling_ratios: {table: x, layout: y}`로 조정 가능.

### 5-4. 프롬프트 구성 (`utils/prompt_templates.py`)
한 샘플은 `system + user + assistant` 3-turn 대화로 만들어진다.

- **system**: 통합 스타일이면 `UNIFIED_SYSTEM_PROMPT`
  (“문서 이미지를 분석해 element들의 JSON 배열만 출력하라”).
- **user**: `prompt_style`에 따라 분기.
  - `unified_layout`: "페이지의 모든 layout element를 찾아 JSON 배열로 내라" + 출력 스펙 + 허용 라벨 목록.
  - `unified_table_with_ocr`: "표 1개를 category=Table 단일 원소 배열로 내라" + **OCR 목록 포함** + 출력 스펙.
  - `unified_table_without_ocr`: 위와 같지만 OCR 없이.
- **assistant(정답)**: `gt_html`(통합 JSON 문자열). `include_thinking: false`라 `<think>`는 안 붙인다.

### 5-5. 배치 생성 + 손실 마스킹 (`train/collator.py`의 `MultimodalCollator`)
- 위 대화 + 이미지를 토크나이저/프로세서로 토큰화.
- **loss는 assistant(정답) 토큰에만** 계산한다. system/user/이미지 토큰은 `-100`(IGNORE_INDEX)으로
  마스킹 → 모델이 "정답을 생성하는 능력"만 학습한다.
- 통합 레코드는 `gt_html`과 `prompt_style`을 그대로 읽으므로 **collator는 수정 불필요**.

### 5-6. LoRA 학습 루프
- 4-bit 양자화된 9B 모델은 고정, **LoRA 어댑터만 학습**.
- `num_train_epochs`, `learning_rate(1.5e-5)`, `cosine` 스케줄, gradient checkpointing 등은 config 참조.
- `save_steps: 100`마다 **체크포인트(어댑터) 저장**, `eval_steps: 100`마다 검증 손실(`eval_loss`) 측정.
- `output/unified_smoke/student_sft/checkpoint-XXX/` 형태로 누적(`save_total_limit: 3`).
- early stopping은 `eval_loss` 기준.

> 학습 중 자동으로 찍히는 성능 지표는 **`eval_loss`(검증 손실)뿐**이다. 이는 "정답 토큰을 얼마나
> 잘 맞히나"의 대리지표일 뿐, TEDS나 레이아웃 F1 같은 **태스크 성능은 따로 평가(6장)** 해야 한다.

---

## 6. 평가(성능 측정) — 새로 만든 통합 지표

기존 `eval/evaluate.py`는 모델 출력이 "표 HTML 그대로"라고 가정해서, **JSON 배열을 내는
통합 모델에는 못 쓴다.** 그래서 통합 출력 전용 평가를 새로 만들었다.

### 6-1. 왜 새 지표가 필요한가
- 통합 출력은 `JSON 배열`이라 먼저 **파싱**해야 한다.
- 한 데이터셋 안에 **테이블 샘플 + 레이아웃 샘플**이 섞여 있어, 둘을 **다른 지표**로 봐야 한다.

### 6-2. 지표 설계 (최적 조합)

**(공통) JSON 파싱 성공률 `json_parse_success_rate`**
- 모델 출력이 우리가 정한 JSON 배열 스키마로 파싱되는 비율(0~1).
- 출력 "형식" 자체를 잘 배웠는지 보는 1차 관문. 낮으면 다른 지표는 의미가 없다.

**(테이블 샘플) — 기존과 동일 지표라 e18 등 과거 모델과 직접 비교 가능**
- `avg_teds`: Tree-Edit-Distance Similarity. **표 구조 + 셀 내용** 종합 유사도(0~1, 높을수록 좋음).
- `avg_teds_structure`: 셀 내용 빼고 **구조(행/열/병합)만** 비교.
- `avg_span_f1`: 병합셀(colspan/rowspan) 검출 F1.
- 계산: 예측 JSON에서 `category=="Table"` 원소의 `text`(HTML)를 꺼내 정답 HTML과 비교.
  내부적으로 기존 `eval/metrics.py`의 TEDS/Span 로직을 그대로 재사용.

**(레이아웃 샘플) — 객체 검출에서 표준적이고 해석이 쉬운 지표**
- `avg_layout_f1`: **element-level F1**. 예측 영역과 정답 영역을 매칭한다.
  - 매칭 규칙(TP): **bbox IoU ≥ 0.5 그리고 category 일치**.
  - IoU = 두 네모의 (겹친 넓이 / 합친 넓이). 1에 가까울수록 위치가 정확.
  - greedy 매칭(IoU 큰 쌍부터 1:1 매칭), precision/recall로 F1 계산.
- `avg_layout_precision` / `avg_layout_recall`: 각각 "예측이 헛돌지 않았나 / 정답을 놓치지 않았나".
- `avg_layout_category_accuracy`: **위치(IoU)만으로 매칭**한 쌍에서 category가 맞은 비율
  (위치는 맞췄는데 종류를 틀렸는지 따로 본다).
- `avg_layout_mean_iou`: 매칭쌍 평균 IoU(위치 정확도).
- `avg_layout_text_sim`: 매칭쌍의 텍스트 유사도(정규화 편집거리 기반, 1에 가까울수록 동일).

> 왜 mAP가 아니라 F1@IoU0.5인가: 레이아웃 카테고리가 다양하고 첫 학습 규모가 작아,
> 신뢰도 점수 기반 mAP보다 **고정 임계값 F1 + 보조 지표(IoU/category/text)**가 더 안정적이고
> 팀원이 해석하기 쉽다. IoU 임계값은 `--iou_threshold`로 바꿀 수 있다.

### 6-3. 산출물
`eval/evaluate_unified.py` 실행 시 `--output_dir`에 다음이 생긴다.
- `metrics_unified.json`: 위 모든 지표의 집계값(테이블/레이아웃 블록 분리).
- `predictions_unified.jsonl`: 샘플별 예측·정답·지표·원응답(디버깅용).

`metrics_unified.json` 예시:
```json
{
  "total_samples": 200, "table_samples": 180, "layout_samples": 20,
  "json_parse_success_rate": 0.98,
  "table":  {"avg_teds": 0.71, "avg_teds_structure": 0.83, "avg_span_f1": 0.55, ...},
  "layout": {"avg_layout_f1": 0.42, "avg_layout_precision": 0.50,
             "avg_layout_recall": 0.37, "avg_layout_category_accuracy": 0.80,
             "avg_layout_mean_iou": 0.66, "avg_layout_text_sim": 0.61, ...},
  "avg_inference_time": 6.4, "backend": "api", "iou_threshold": 0.5
}
```

### 6-4. 추론 백엔드
`evaluate_unified.py`는 세 가지를 지원한다(메시지의 system/user 프롬프트는 학습과 동일하게 통합 스타일 사용).
- `api`(기본): OpenAI 호환 엔드포인트. 서버 vLLM 서빙(`http://192.168.75.173:8010`) 평가.
- `vllm`: 로컬 vLLM. **LoRA 어댑터 경로**를 주면 base 모델 위에 얹어 평가.
- `transformers`: 로컬 4-bit + PeftModel.

---

## 7. 변경/추가된 파일

### 신규
- **`data/build_unified_dataset.py`** — 테이블/레이아웃 JSONL → 통합 스키마 변환·병합·split.
- **`config/exp_unified_smoke.yaml`** — 통합 학습 설정(task 1:1, JSON 출력, LoRA).
- **`scripts/train_unified.sh`** — 통합 학습 실행.
- **`eval/unified_metrics.py`** — 통합 출력 파싱 + 테이블(TEDS/Span) + 레이아웃(element F1 등) 지표.
- **`eval/evaluate_unified.py`** — 통합 모델 추론 + 지표 집계 메인 스크립트.
- **`scripts/eval_unified.sh`** — 통합 평가 실행.
- **`docs/UNIFIED_TABLE_LAYOUT_PIPELINE.md`** — 본 문서.

### 수정
- **`utils/prompt_templates.py`**
  - 스타일 추가: `unified_table_with_ocr`, `unified_table_without_ocr`, `unified_layout`.
  - `UNIFIED_SYSTEM_PROMPT` + JSON 출력 스펙/라벨 템플릿.
  - `get_user_prompt_with_style`/`build_chat_messages`에 통합 분기, `is_unified_style()` 헬퍼.
- **`train/train_qlora.py`** (`TSRDataset`)
  - `task_type` 보존, 레이아웃은 `fill_empty_cells`/`ocr_mix` 대상에서 제외.
  - `get_task_counts()`, `get_task_sampling_weights()` 추가(task 1:1).
- **`distill/student_sft.py`**
  - `task_mix_sampling` 옵션 → task 기반 `WeightedRandomSampler`(Subset도 지원).

> `train/collator.py`는 `gt_html`을 정답, `prompt_style`을 레코드에서 읽으므로
> 통합 레코드를 **수정 없이** 처리한다.

---

## 8. 실행 명령어 모음

### (1) 통합 데이터셋 빌드
```bash
cd train/vlm
python -m data.build_unified_dataset \
  --table-dir    ./_train_data/table_src_6902 \
  --layout-jsonl /NHNHOME/WORKSPACE/0426030039_A/jhshin/layout_export/layout_from_existing79.jsonl \
  --layout-image-root <레이아웃 이미지 루트> \
  --out-dir ./_train_data/unified_smoke \
  --bbox-scale 1024
# 옵션: --copy-images (이미지 복사), --max-table-per-split N (스모크용 상한)
```
산출물:
```
_train_data/unified_smoke/
  data/{train,valid,test}.jsonl   # 테이블 + 레이아웃 통합
  build_summary.json              # split별 table/layout 개수
  (images/  ← --copy-images 시)
```

### (2) 학습
```bash
cd train/vlm
# config(student.name_or_path)의 모델 경로가 학습 서버에 맞는지 먼저 확인
NUM_GPUS=4 bash scripts/train_unified.sh
# 체크포인트: output/unified_smoke/student_sft/checkpoint-XXX/
```

### (3) 평가
```bash
cd train/vlm

# (A) 서버 API로 서빙 중인 통합 모델 평가
API_URL=http://192.168.75.173:8010/v1/chat/completions \
API_MODEL=<served-model-name> \
TEST_DATA=_train_data/unified_smoke/data/test.jsonl \
bash scripts/eval_unified.sh

# (B) 학습으로 나온 LoRA 어댑터를 로컬 vLLM으로 평가
BACKEND=vllm \
MODEL=output/unified_smoke/student_sft/checkpoint-300 \
BASE_MODEL=/home/vlm_train/models/Qwen3.5-9B \
TEST_DATA=_train_data/unified_smoke/data/test.jsonl \
bash scripts/eval_unified.sh

# 결과: eval_results/unified_*/metrics_unified.json, predictions_unified.jsonl
```

---

## 8-2. 출력 형식 파이프라인 (JSON ↔ HTML)

통합 모델은 **두 가지 출력 형식**을 지원한다. 동일한 SFT+LoRA 학습 방식을 유지하며,
설정만 바꿔 파이프라인을 전환한다.

| 항목 | JSON 파이프라인 (기존) | HTML 파이프라인 (신규) |
|------|----------------------|----------------------|
| 정답(table) | `[{"bbox","category":"Table","text":표HTML}]` | 표 HTML 그대로(변형 없음) |
| 정답(layout) | element JSON 배열 | reading-order HTML 조각 |
| prompt_style | `unified_*` | `unified_html_*` |
| system prompt | `UNIFIED_SYSTEM_PROMPT` | `UNIFIED_HTML_SYSTEM_PROMPT` |
| 표 평가 | JSON 파싱 → TEDS | HTML 직접 → TEDS |
| 레이아웃 평가 | IoU 기반 element F1 | **HTML 트리 TEDS**(bbox 없음 → IoU 불가) |
| 학습 config | `config/exp_unified_smoke.yaml` | `config/exp_unified_html_smoke.yaml` |

**레이아웃 → HTML 변환 규칙** (`utils/html_unified.py`):
- 래퍼 태그(`<!DOCTYPE>/<html>/<head>/<body>`) 없음 — 표 데이터와 동일한 형태.
- 표 데이터가 쓰던 태그 우선(`table/tr/td/th/p/span/...`), 레이아웃 전용은 추가 태그 허용.
- 매핑: Title→`<h1>`, Section-header→`<h2>`, Text/Caption/Footnote/Formula/Page-header/Page-footer→`<p>`,
  List-item→`<li>`(연속 항목을 하나의 `<ul>`로 묶음), Picture→`<img/>`, Table→표 HTML 그대로.
- text 앞의 markdown heading prefix(`#`,`##`,...)는 제거. 줄바꿈은 `<br/>`.

### HTML 데이터셋 만들기
방법 A — 기존 JSON 셋을 변환(권장: 이미지/split 재사용, 공정 비교):
```bash
cd train/vlm
python -m data.convert_unified_json_to_html \
  --src-dir ./_train_data/unified_smoke \
  --out-dir ./_train_data/unified_html_smoke
# images/ 는 원본으로 심볼릭 링크됨
```
방법 B — 원본 소스에서 직접 빌드:
```bash
python -m data.build_unified_dataset ... --output-format html \
  --out-dir ./_train_data/unified_html_smoke --copy-images
```

### HTML 파이프라인 학습 / 평가
```bash
# 학습 (HTML)
CONFIG=config/exp_unified_html_smoke.yaml \
NUM_GPUS=4 bash scripts/train_unified.sh

# 평가 (HTML) — output_format 은 TEST_DATA 의 prompt_style 로 자동 감지
TEST_DATA=_train_data/unified_html_smoke/data/test.jsonl \
API_URL=http://192.168.75.173:8010/v1/chat/completions \
API_MODEL=<served-model-name> \
bash scripts/eval_unified.sh
```
HTML 평가 결과 메트릭(`metrics_unified.json`)은 `output_format: "html"`,
`html_parse_success_rate`, table의 `avg_teds`, layout의 `avg_layout_teds`를 포함한다.

---

## 9. 주의 / 알려진 제약

- **`train_ocr_mix_ratio` 미사용**: 통합에서는 `prompt_style`을 빌드 단계에서 확정한다.
  켜면 테이블 prompt_style이 chandra로 바뀌므로 **설정 금지**(켜져도 레이아웃은 가드됨).
- **`annotate_empty_cells: false` 권장**: 빈 셀 채우기는 표 HTML 전용. 통합 JSON 정답에는
  적용하지 않는다(레이아웃은 코드에서 가드).
- **레이아웃 규모(~80)**: 현재는 스모크/첫 학습용. 본 학습은 레이아웃 수집(~1만) 완료 후 재빌드 필요.
- **모델 경로 확인**: `config/exp_unified_smoke.yaml`의 `student.name_or_path`를 학습 서버
  (174 또는 DCTN)에 맞게 확인.
- **레이아웃 bbox 정규화**: 원본 dots.mocr bbox는 픽셀 좌표라 빌드 시 이미지 크기로
  0~`bbox_scale` 정규화한다. 이미지 로드 실패 레코드는 제외(`build_summary.json`의 `build_stats`).
- **평가 의존성**: 테이블 TEDS 계산은 `editdistance`, `apted`, `beautifulsoup4`, `lxml` 필요.
  없으면 `uv pip install editdistance apted beautifulsoup4 lxml` (학습 서버 venv 기준).
- **평가 프롬프트 일치**: `evaluate_unified.py`는 레코드의 `prompt_style`/`task_type`으로
  학습과 동일한 통합 프롬프트를 재구성한다(테이블은 OCR 포함). 학습-평가 입력 형태를 맞추기 위함.

---

## 10. 용어집 (빠른 참고)

| 용어 | 뜻 |
|------|----|
| VLM | 이미지+텍스트를 입력받아 텍스트를 내는 모델(Qwen3-VL) |
| TSR | 표 이미지를 HTML 구조로 복원하는 작업 |
| 레이아웃 인식 | 페이지에서 제목/본문/표/그림 영역을 찾는 작업 |
| LoRA | 원본 모델은 고정하고 작은 보조 행렬만 학습하는 파인튜닝 |
| QLoRA | 4-bit 양자화 + LoRA (메모리 절약) |
| bbox | `[x0,y0,x1,y1]` 네모 영역 좌표 |
| IoU | 두 네모의 겹친넓이/합친넓이 (위치 정확도, 0~1) |
| TEDS | 표 트리 편집거리 기반 유사도 (표 정확도, 0~1) |
| Span | 병합셀 (colspan/rowspan) |
| F1 | precision·recall의 조화평균 |
| loss masking | 정답(assistant) 토큰에만 손실을 줘서 생성 능력만 학습 |
| eval_loss | 검증셋의 평균 손실. 학습 중 자동 기록되는 유일한 지표 |
| 체크포인트 | 학습 도중 저장한 (LoRA 어댑터) 스냅샷 |
