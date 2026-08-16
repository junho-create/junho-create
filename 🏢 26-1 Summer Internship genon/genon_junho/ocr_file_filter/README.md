# ocr_file_filter

VL 문서 파싱 데이터셋 큐레이션 엔진 (MinerU2.5-Pro 방식). 흔한 데이터는 버리고 **롱테일·어려운
페이지**만 골라 타겟 모델(Qwen3.5-9B, Chandra div-HTML 포맷)의 약점을 자동으로 SFT 데이터로 만든다.
최종 목적지는 이 모델을 계속 SFT하는 것 — 그래서 마지막 산출물은 구조화 라벨이 아니라 그 모델이
이미 학습된 것과 동일한 `gt_html` 문자열 포맷이다(아래 [7] 참고).

## 파이프라인 한눈에

```
[0]   io        원본(PDF/이미지) → 통일 스키마 (unified.jsonl)
[1~3] cluster/taster/ddas   (선택) ViT 임베딩 클러스터링 → 난이도 추정 → 예산 기반 샘플 선정
[4]   cmcv      3모델 교차검증(layout/text/table/formula subtask 별) → Easy/Medium/Hard 티어
[4.5] rescue    Hard 페이지에서 요소(bbox) 단위로 두 외부 모델 합의분만 구제
[5]   report    갤러리 HTML 검수 (선택)
[6]   hardcase  Rescue로도 안 풀린 Hard만 — heavy judge 모델이 생성+render-then-verify 판정
[merge]         cmcv(Medium) + rescue(Medium-rescued) + hardcase(Hard resolved) → final_dataset.jsonl
[7]   export    최종 라벨 → SFT 학습 포맷 gt_html 변환 (filtering_result/gt_html_dataset.jsonl)
```

- **타겟(파인튜닝 대상):** Qwen3.5-9B + LoRA merge (Chandra div-HTML 포맷)
- **CMCV 외부 2종:** `dots.ocr` · `PaddleOCR-VL 1.6`(실제 PaddleOCRVL 파이프라인, 배치 사전계산)
- **hardcase judge:** 현재 `Qwen3.5-122B-A10B-FP8`(PP=3) — 아래 "judge 모델 선택" 참고
- **CDM(수식 채점, Formula subtask 전용):** 외부 서버/모델 없음 — 로컬 TeX Live 렌더링(아래 CDM 설정 참고)
- 전부 vLLM OpenAI-호환 엔드포인트. 서빙/GPU 배정 상세는 `configs/models.yaml`.

이 문서는 **왜 이렇게 자동 레이블이 매겨지는지(로직)** 를 중심으로 설명한다. 스텝별 실측 트러블슈팅
(워커 수, GPU 경합, 임계값 재보정 등)은 `PIPELINE.md`에 있다.

---

## 핵심 로직 — Hard 케이스는 어떻게 자동으로 라벨링되는가

Hard = "target(9B)·dots.ocr·PaddleOCR-VL 세 모델이 서로 다 불일치하는 페이지/블록" — 즉 아무도
믿을 근거가 없는 케이스다. 여기에 순서대로 세 번의 그물을 쳐서 최대한 사람 손 없이 GT를 만든다.

### 1) CMCV — 티어 판정 로직 (`ocr_filter/cmcv/run.py`)

한 페이지를 `layout / text / table / formula` 4개 **subtask**로 쪼개서 **subtask마다 따로** 티어를
매긴다(페이지 전체를 하나로 뭉뚱그리면 "본문은 멀쩡한데 표 하나 때문에 페이지 전체가 Hard로
버려지는" 손실이 컸음 — 그래서 subtask 분리가 필요했고, 이 손실을 나중에 되찾는 게 rescue 단계).

subtask마다 세 쌍(`target-dots`, `target-paddle`, `dots-paddle`)의 일치도 점수를 내고, **`agree_min`
(기본 0.85) 하나로 이진 판정**한다:

```python
def tier(td, tp, dp, agree_min=0.85):
    if td >= agree_min or tp >= agree_min:
        return "Easy"      # target 이 dots.ocr / paddle 중 최소 하나와 일치
    if dp >= agree_min:
        return "Medium"    # target 은 둘 다와 불일치, 외부 둘끼리는 일치
                            #  → dots.ocr 출력을 pseudo-label 로 채택 (SFT 핵심)
    return "Hard"           # 세 쌍 다 불일치 → 자동 라벨링 불가
```

평균이 아니라 "최소 하나 일치"를 보는 이유: 평균을 쓰면 "하나랑만 잘 맞고 나머지는 안 맞는" Easy와
"외부 둘만 잘 맞는" Medium이 둘 다 뭉개져 Hard로 오분류된다.

일치도 점수 자체는 subtask별로 다른 지표를 쓴다(논문 MinerU2.5-Pro §3.2 방식):

| subtask | 지표 | 비고 |
|---|---|---|
| layout | PageIoU | bbox 배치(커버리지) 자체의 일치도 |
| text | edit distance | Table/Formula 요소는 제외한 본문 평문 |
| table | TEDS | bbox로 매칭된 표끼리만(순서 매칭 아님, 아래 참고) |
| formula | **CDM** | bbox로 매칭된 수식끼리만 — 아래 별도 설명 |

**표/수식은 등장 순서가 아니라 bbox IoU로 매칭한다** — 모델마다 표를 1~2개로 다르게 잡는 경우가
흔해서 순서 매칭을 쓰면 한쪽이 요소 하나만 더/덜 잡아도 뒤가 전부 밀려 오답 처리된다(실측: Table
슬롯 중앙값이 정확히 0.000으로 나온 원인). 한쪽에만 있는(짝 없는) 표/수식은 **0점**으로 집계해서
"통째로 누락/환각"도 벌점을 받게 한다.

**Formula subtask(CDM)만 특이한 점**: LaTeX 두 개를 실제로 `pdflatex`+`ImageMagick`으로 렌더링해서
문자 단위로 시각적 매칭한 F1 스코어다(`ocr_filter/metrics/cdm.py`) — 편집거리가 아니라 "그려보면
같아 보이는가"를 본다. 렌더 자체가 실패하면(TeX 미설치, 폰트 문제 등) 가짜 0점을 주지 않으려고
`None`을 돌려주는데, `None`인 subtask는 `_page_tier`가 "해당 없음"으로 제외해버린다. 그래서 **CDM
환경이 죽어 있으면 수식이 있는 페이지가 검증 없이 Easy/Medium으로 새어나갈 수 있다** — 이 구멍은
merge 단계(`_has_unverified_formula`, 아래 참고)에서 다시 걸러낸다. 같은 프로세스에서 렌더가 연속
8회 실패하면 서킷브레이커가 작동해 그 뒤로는 호출 없이 계속 `None`을 준다(수 초짜리 pdflatex 호출을
계속 낭비하지 않기 위함) — CDM이 계속 죽어있다 싶으면 `scripts/setup_cdm_texlive.sh`부터 확인.

Medium의 pseudo-label(dots.ocr 채택)은 블렌드 점수(`agree_min`) 통과 **외에** 텍스트 전용 게이트
(`text_min=0.90`)도 따로 통과해야 한다 — 블렌드에는 레이아웃 전용 슬롯(PageIoU 등)이 절반 넘게
섞여 있어서 "박스 위치는 같은데 글자를 다르게 읽은" 쌍이 agree_min을 넘길 수 있기 때문. 게이트에
걸리면 그 subtask는 Hard로 강등된다.

### 2) Rescue — 요소 단위 구제 (`ocr_filter/cmcv/rescue.py`, GPU 안 씀)

CMCV까지만 돌리면 표/수식 하나 때문에 페이지의 본문 전체가 Hard로 버려지는 페이지가 많다(실측:
페이지 대표 티어 Hard 40%인데 그중 상당수는 text subtask는 Easy/Medium). Rescue는 그 본문을
되찾는다:

1. dots.ocr·paddle 두 **외부** 모델 출력에서 겹치는 bbox끼리 **N:M 그룹**으로 묶는다(1:1 매칭이
   아니라 그룹인 이유: dots가 문단 하나로 묶는 블록을 paddle이 줄 단위로 쪼개는 식의 분할 관성
   차이 때문에 1:1 IoU 매칭 회수율이 1%대에 그쳤고, 그룹으로 묶어 텍스트를 이어붙이면 22배 뛴다).
2. Picture/Table/Formula/Section-header는 그룹 내에서도 **엄격 1:1 매칭·합의**를 요구(구조를
   정의하는 요소라 갈리면 안 됨). 나머지 본문류는 카테고리가 달라도(List-item vs Text 등) 텍스트
   풀 전체를 이어붙여 합의 판정.
3. 합의된 요소만 채택 — category 태그는 항상 dots.ocr 기준(표준 11종 스키마), Table의 콘텐츠만
   paddle 채택(셀 구조가 더 안정적).
4. 합의된 요소가 페이지 전체 영역의 `coverage_min`(기본 0.90) 이상을 못 덮으면 rescue 실패로
   처리하고 다음 단계(hardcase judge)로 넘긴다 — 라벨이 듬성듬성한 페이지는 후처리 필터에서
   어차피 탈락하므로 여기서 미리 끊는다.

모델 재호출이 전혀 없어서(이미 저장된 cmcv 결과만 재사용) CMCV가 GPU를 쓰는 동안 CPU에서 병행
가능하다.

### 3) Hardcase judge — 197B/122B가 스스로 만들고 스스로 심사 (`ocr_filter/hardcase/`)

Rescue로도 못 건진 순수 Hard만 대상. 되먹임 재교정 루프(옛 방식)는 안 바뀔 bbox를 큰 모델로 계속
재확인하는 낭비가 커서 폐기하고, **단일 콜 생성 + 판정 1회(+필요시 교정 1회)** 로 바꿨다:

1. **생성**: heavy judge 모델에게 원본 이미지 + dots.ocr/paddle 예측 두 개를 레퍼런스로 주고
   Chandra div-HTML 포맷으로 최종 라벨을 한 번에 생성.
2. **위생 검사**: bbox 없는 요소뿐이거나 텍스트 카테고리 절반 이상이 빈 텍스트면 judge에 보내지도
   않고 바로 사람 검수행 — 빈 라벨을 judge가 "이상 없음"으로 통과시키는 사고를 사전 차단.
3. **판정(render-then-verify)**: 원본 + bbox 오버레이뿐 아니라, **파싱된 표 HTML/수식 LaTeX를
   실제로 렌더링한 스크린샷**까지 최대 4장을 같이 보여주고 PASS/FAIL 판정한다. 오버레이만 보면
   박스 위치만 검증 가능하고 표/수식의 **내용·구조**는 원리적으로 검증이 안 된다 — 실제로 그
   상태로 대량 케이스가 텍스트 무검증인 채 확정됐던 걸 발견한 뒤 도입(MinerU2.5-Pro §3.3 방식).
4. FAIL이고 정확한 element 인덱스 지적(`element_issues`)이 있으면 **딱 한 번만** 교정 후
   재판정(루프 아님).
5. 최종 PASS(`resolved=True`)면 자동 GT 확정, FAIL이면 그 라벨을 초안(prelabel) 삼아
   `hardcase/export_labeler.py`를 통해 사람 검수 UI(`labeler/`)로 넘어간다.

### judge 모델 선택 — 397B가 이상적, 122B는 서버 제약으로 대체 중

논리상 judge/생성 품질은 모델이 클수록(판정 신뢰도·render-then-verify에서 미묘한 결함을 잡아내는
능력 모두) 좋아진다. 원래 목표는 **Qwen3.5-397B-A17B**였는데, 이 서버(H100×4, 그중 실사용 가능한
건 3장=240GB)에는 397B(NVFP4, 251GB, B200 전용 커널)가 물리적으로 안 들어가서 지금은
`Qwen3.5-122B-A10B-FP8`(127GB, PP=3, GPU 0/2/3 전부 사용)로 대체해 쓰고 있다(`configs/models.yaml`
`judge` 섹션). **GPU 여유가 더 되는 서버(B200 등)를 쓸 수 있으면 397B로 바꾸는 걸 권장** — 특히
render-then-verify 판정 정확도와 Hard 케이스 생성 품질에 직접 영향을 준다.

### 4) 병합 (`scripts/build_final_output.py`)

세 소스를 id 기준으로 합쳐 `final_dataset.jsonl`을 만든다:

- **Medium**: cmcv의 pseudo-label(dots.ocr 원본) 채택. `subtask_tiers.formula`가 전부 `None`인데
  target 출력에 Formula 요소가 있는 경우(=CDM이 죽어서 검증을 못 한 경우)는 여기서 걸러내
  hardcase 재검증으로 돌린다(`_has_unverified_formula`).
- **Medium-rescued**: rescue가 구제한 요소 단위 라벨.
- **Hard**: hardcase judge에서 `resolved=True`인 것만("resolved=False"는 최종 산출물에 안 들어가고
  사람 검수 큐에 남는다).
- Easy는 target(9B) 자기 출력이라 SFT 가치가 낮아 여기서 아예 안 받는다.

---

## 실행 방법 (두 가지 버전)

### A. 전체 데이터 자동 라벨링 — 클러스터 없이 (권장, 신규 배치는 보통 이걸 씀)

원본 이미지 폴더 하나 전체에 대해 **샘플링 없이** cmcv~hardcase judge~export까지 끝(`final_gt.jsonl`)
까지 한 번에 돈다. 클러스터/taster/ddas 예산 배분 단계를 아예 안 거치는 이유는 — 난이도 기반으로
일부만 뽑는 게 아니라 **배치 전체를 다 라벨링해서 뽑아내려는 목적**이기 때문이다(신규 배치는
전량이 SFT 후보).

H100 서버에서 `scripts/run_pipeline.sh`만 실행시키면 된다:

```bash
cp configs/batch5400.yaml configs/<새배치>.yaml   # paths.work_dir 만 새 경로로 수정
scripts/run_pipeline.sh configs/<새배치>.yaml <원본이미지_디렉터리> [max_uncovered_new]
```

각 단계는 대부분 resume-safe(이미 처리한 id는 건너뜀)라 중간에 죽어도 그냥 다시 실행하면 이어서
간다. GPU 3장(H100 80GB) 필요, CMCV 3모델과 judge가 GPU를 나눠 쓰므로 스크립트 안에서 순차로
내리고 올린다. **워커 수·GPU 경합·임계값 재보정 등 실측 함정은 전부 `PIPELINE.md`에 정리되어
있으니 막히면 거기부터 볼 것.**

### B. 클러스터 샘플링 버전 — 난이도 기반으로 일부만 뽑을 때

전체를 다 처리하기엔 너무 크거나(예: 6만 장대), 예산을 정해두고 어려운 데이터 위주로만 뽑고 싶을
때 쓴다. ViT 임베딩으로 클러스터링 → 클러스터별 소량 CMCV로 난이도 추정(taster) → 난이도 기반
예산 배분(ddas)으로 대상 id를 추리고, 그 subset만 본 CMCV~hardcase에 태운다.

```bash
cd /NHNHOME/WORKSPACE/0426030039_A/jhyeo/ocr_file_filter
./venv/bin/python3 -m ocr_filter.cli io build-raw --config configs/new_docs.yaml \
  --input-dir <원본_PDF_디렉터리> --images-out <이미지_출력> --out <work_dir>/unified.jsonl
./venv/bin/python3 -m ocr_filter.cli cluster build --config configs/new_docs.yaml \
  --n-clusters-page 128 --n-clusters-element 0
./venv/bin/python3 -m ocr_filter.cli taster run --config configs/new_docs.yaml --workers 8
./venv/bin/python3 -m ocr_filter.cli ddas select --config configs/new_docs.yaml \
  --ddas-n-total <원하는_전체_예산>
./venv/bin/python3 -m ocr_filter.cli cmcv run --config configs/new_docs.yaml \
  --ids-file <work_dir>/ddas_selected_ids.txt --workers 8
./venv/bin/python3 -m ocr_filter.cli rescue run --config configs/new_docs.yaml
./venv/bin/python3 -m ocr_filter.cli hardcase run --config configs/new_docs.yaml
./venv/bin/python3 -m ocr_filter.cli hardcase export --config configs/new_docs.yaml
# → <work_dir>/labeler_export/review.html (사람 검수)
```

두 버전 모두 최종적으로 `scripts/build_final_output.py`(병합) → `export gt-html`(아래 [7]) 순서로
끝난다.

---

## 서버 고정 경로

| 용도 | 경로 |
|---|---|
| 코드 (이 repo) | `/NHNHOME/WORKSPACE/0426030039_A/jhyeo/ocr_file_filter` |
| 실행 파이썬 (반드시 이걸로) | `.../ocr_file_filter/venv/bin/python3` |
| 모델 weight | `.../ocr_file_filter/_models` (259GB, git 제외) |
| 2차(GT 없는 신규 배치) 산출물 | `.../jhyeo/ocr_filter_result` (work_dir, jhyeo/ 밑에 자체완결) |
| 최종 SFT 산출물 | `.../ocr_file_filter/filtering_result/gt_html_dataset.jsonl` |
| 기존 레퍼런스 학습셋(16,886건, gt_html) | `.../jhyeo/jhyeo_trash/mineru_work/finetune/chandra_table_layout_divhtml_16886` — 최종 SFT 때 위 산출물과 합쳐 넣는 대상 |

`--config` 미지정 → `configs/default.yaml`. 새 배치는 `configs/batch5400.yaml`(또는 최신 배치
config)을 복사해 `paths.work_dir`만 바꿔서 쓴다.

## 서버 기동 (공통, 최초 1회)

```bash
cd /NHNHOME/WORKSPACE/0426030039_A/jhyeo/ocr_file_filter
./venv/bin/python3 -m ocr_filter.cli models serve    # target/dots.ocr/paddle/judge (백그라운드)
./venv/bin/python3 -m ocr_filter.cli models status   # UP/DOWN 헬스체크
```
> 새 서버에 처음 올릴 땐 `models download`(→`_models`) 먼저, 폐쇄망이면 `models bundle out.tar.gz`
> 로 옮긴다. 상세는 `configs/models.yaml`. `scripts/run_pipeline.sh`를 쓰면 CMCV↔judge 서버
> 기동/종료를 스크립트가 순서대로 알아서 해준다(수동으로 `models serve/stop` 안 해도 됨).

## [7] SFT 학습 포맷 변환 (export gt-html)

`final_dataset.jsonl`(또는 `final_gt.jsonl`)의 `label`(`[{"category","text","bbox"}, ...]`)은 아직
실제 학습 포맷이 아니다 — 파인튜닝 대상(target)은 이미지당 하나의 **`gt_html` 문자열**
(`<div data-bbox="x0 y0 x1 y1" data-label="Category"><p>...</p></div>` 연결)로 학습돼 있고, 새
데이터도 기존 레퍼런스 학습셋과 **같은 SFT 런에 섞이므로** 정확히 같은 포맷이어야 한다.
`resolved==True`인 레코드만(Hard 최종 실패분 제외) tier별 bbox 좌표계(Easy=이미 0-1000,
Medium/Hard=원본 픽셀 → 이미지 크기로 정규화)를 통일해 변환한다. 자세한 설계는
`ocr_filter/export/gt_html.py` 모듈 docstring 참고.

```bash
./venv/bin/python3 -m ocr_filter.cli export gt-html
```

출력 스키마(레퍼런스 학습셋과 동일): `{"image_path", "gt_html", "prompt_style": "unified_layout",
"bbox_scale": 1000, "output_format": "html", "ocr_info": []}`.

### 카테고리 레이블 정규화

hardcase judge(대형 모델) 출력은 프롬프트로 표준 11종 레이블을 지시해도 그 스키마로 파인튜닝된 적
없는 모델이라 잘 안 지켜진다(실측 13%가 비표준 — 케이스 변형 + `chart`/`reference_content` 등
자유형식). `ocr_filter/cmcv/normalize.py`의 `normalize_category()`가 모든 파서 출력에 적용돼
케이스/구두점 차이는 표준으로, 자유형식은 가장 가까운 표준으로, 나머지는 `Text`로 폴백한다.

## CDM(수식 채점) 설정 — cmcv의 Formula subtask 채점용

`ocr_filter/metrics/cdm.py`가 LaTeX 두 개를 각각 렌더링해서 문자 단위로 매칭하는 CDM 스코어러다.
렌더링에 TeX Live + ImageMagick이 필요한데, 이 프로젝트 **자체 유저 디렉토리**(`_texlive/`, git
제외)에 설치해서 쓰므로 최초 1회만 실행하면 된다(root 불필요, 시스템에 이미 pdflatex/magick이
있으면 그쪽으로 자동 폴백):

```bash
bash scripts/setup_cdm_texlive.sh   # 수 GB, 수십 분 걸림. _texlive/ 있으면 스킵(멱등)
```

CDM 없이도 cmcv는 정상 동작한다 — 렌더 환경이 없으면 `cdm_score()`가 `None`을 반환해 Formula
슬롯만 조용히 빠지고(가짜 점수 방지), 나머지 채점(text/table/layout)은 그대로 진행된다. 단, 위
"핵심 로직" 절에서 설명한 대로 CDM이 죽어 있으면 수식 검증이 안 된 채로 새어나갈 수 있으니 새
배치를 돌리기 전에 CDM이 살아있는지 먼저 확인할 것.

## 다른 서버로 이전하기

**가져가야 할 것** (`jhyeo/` 밑 통째로, 아래 "빼고 가는 것"만 제외):
- `jhyeo/ocr_file_filter` — 코드 전체(이 repo)
- `jhyeo/ocr_filter_result` — 신규 배치 산출물 전체(raw/images/unified.jsonl/cmcv_results.jsonl/
  rescue_results.jsonl/hardcase_judge*.jsonl/final_gt.jsonl). jhyeo/ 밑에 자체완결 — 다른 사용자
  디렉토리 참조 없음.
- `jhyeo/jhyeo_trash/mineru_work/finetune/chandra_table_layout_divhtml_16886` — 최종 SFT 때
  합쳐 넣을 기존 레퍼런스 학습셋(이미지 경로가 상대경로라 그 자체로 이식 가능).

**빼고 가는 것** (`.gitignore` 처리됨, 새 서버에서 재생성):
- `ocr_file_filter/_models/` (259GB 모델 weight)
- `ocr_file_filter/venv/`
- `ocr_file_filter/_texlive/`, `_texlive_installer/` (CDM용)
- `ocr_file_filter/_work/*.png`, `*.html` 등 캐시성 산출물

**새 서버에서 재생성:**
```bash
cd jhyeo/ocr_file_filter
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt   # + torch/vllm 등은 GPU/CUDA에 맞춰 별도
./venv/bin/python3 -m ocr_filter.cli models download   # 또는 models bundle 로 폐쇄망 이동
bash scripts/setup_cdm_texlive.sh                       # CDM(수식 채점) 필요시만
```

**경로 관련 주의:**
- `configs/*.yaml`의 `paths.*`는 절대경로 하드코딩이 섞여 있다 — 새 서버에서 마운트 경로 자체가
  바뀌면 이 부분들을 찾아 새 경로로 바꿔야 한다. 마운트 경로가 동일하게 유지되면(예: 같은 NFS를
  다른 서버에서도 같은 경로로 마운트) 아무것도 안 바꿔도 된다.
- `configs/models.yaml`은 이 머신의 GPU 배정(포트/gpus/PP·TP 설정)에 맞춰져 있다 — 다른 GPU
  구성(장수·모델·메모리)의 서버로 옮기면 이 파일부터 다시 검증할 것(특히 judge 모델은 위 "judge
  모델 선택" 절 참고 — GPU가 더 넉넉하면 397B로 바꾸는 걸 권장).
