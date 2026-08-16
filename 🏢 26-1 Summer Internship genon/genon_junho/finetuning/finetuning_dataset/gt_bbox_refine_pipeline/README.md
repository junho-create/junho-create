# GT bbox refine pipeline

`combined/train.jsonl`의 layout GT bbox(`gt_html`의 `data-bbox`)를, 같은 레코드에 이미
들어 있는 PaddleOCR 결과(`ocr_info`)를 이용해 후처리로 정밀 보정하는 파이프라인.

**중요: 이건 파이프라인의 맨 마지막 단계다.** `ocr_info`가 이미 채워진 입력이 있어야
돌아간다 — 즉 `add_ocr.py`(PaddleOCR로 OCR 추출해서 `ocr_info`/`bbox_scale`을 붙이는
스크립트, `/home/backup_b200/jhshin/tsr_test/train/vlm/scripts/add_ocr.sh` 참고)가
먼저 끝나서 `train.jsonl`에 `ocr_info`가 있는 상태에서만 실행 의미가 있다.

```
[기존 파이프라인] ... -> add_ocr.py (PaddleOCR 실행, ocr_info 채움) -> train.jsonl
                                                                          |
                                                                          v
                                                        [이 폴더] refine_gt_bbox.py (맨 마지막)
                                                                          |
                                                                          v
                                                                train_refined.jsonl
```

## 배경 / 왜 필요한가

- 기존 GT layout bbox(사람 라벨링 또는 상위 파이프라인 산출)는 줄 단위 위치가 다소
  부정확한 경우가 있다.
- PaddleOCR v5 는 반대로: 글자/짧은 단위 bbox 위치 정확도는 높지만, 텍스트를 누락하거나
  한 줄을 여러 조각으로 과분할하는 경향이 있다.
- 그래서 "PaddleOCR 조각들의 외곽(extent)으로 GT bbox를 스냅 보정"하되, 여러 안전장치로
  오보정(줄 누락으로 인한 축소, 마커/특수문자 잘림 등)을 막는다.

## 파일

- `refine_gt_bbox.py` — 핵심 로직 + CLI. 매칭(포함율+y밴드 1:1 배정) → 보정(`--mode`) →
  가드(마커 보호/교차축 보호/재현율 게이트) 순으로 처리.
- `make_viewer.py` — 정적 HTML 뷰어 생성기. 변경된 박스마다 확대 crop(빨강=기존/파랑=보정)
  을 만들어 self-contained HTML로 저장 (서버 없이 브라우저로 바로 열림).
- `app_refined.py` — 라이브 비교 서버(FastAPI, 포트 8092). 출처(reference_16886 /
  new_63541)를 선택하면 전체 라벨을 회색(기존 GT) + 라벨별 색상(보정 GT)으로 겹쳐 그린
  이미지를 2열 그리드로 보여준다.

## 사용법

### 1. 보정 실행 (train_refined.jsonl 생성)

```bash
cd /home/jhyeo/finetuning/finetuning_dataset/gt_bbox_refine_pipeline
python3 refine_gt_bbox.py \
  --in  /home/jhyeo/finetuning/finetuning_dataset/combined/train.jsonl \
  --out /home/jhyeo/finetuning/finetuning_dataset/combined/train_refined.jsonl \
  --write-out \
  --report report.txt \
  --limit 100000 \
  --mode loose
```

- `--in` 원본은 항상 읽기 전용으로만 쓰인다(수정하지 않음).
- `--mode`: `loose`(권장/최종 채택) / `snap`(보수적, 캡 기반) / `union`(항상 확장만, 텍스트
  잘림 걱정 없지만 뒤틀린 GT는 못 고침). 세 모드의 트레이드오프는 아래 "모드 설명" 참고.
- `--limit` 은 처리할 최대 레코드 수(기본 1000, 전체 처리하려면 데이터 라인 수 이상으로 크게).
- `--include-table` 을 주면 Table 외곽 박스도 보정 대상에 포함(기본은 제외 — 셀 단위 bbox가
  없어서 외곽만 건드리는 게 애매함).
- `--report` 에 결정 통계(채택률, 가드별 기각 사유, 이동량 분포)가 저장됨.

### 2. 정적 뷰어로 빠르게 훑어보기

```bash
python3 make_viewer.py \
  --in /home/jhyeo/finetuning/finetuning_dataset/combined/train.jsonl \
  --out /home/jhyeo/finetuning/finetuning_dataset/combined/train_refined_viewer.html \
  --n 25 --stride 61 --max-crops 10 --mode loose
```

생성된 html은 이미지가 base64로 박혀 있어 **브라우저로 파일만 열면 바로 보임** (서버 불필요).
페이지당 변경폭이 큰 박스 순으로 최대 `--max-crops`개까지 확대 crop해서 보여준다.

### 3. 라이브 비교 서버 (전체 라벨, 회색 vs 라벨색)

```bash
python3 app_refined.py
# -> http://localhost:8092
```

- 위쪽에서 출처(16천장짜리 `reference_16886` / 6만장짜리 `new_63541`)만 고르고 "조회"
  클릭 → 아래 2열 그리드로 이미지가 이어서 나옴(페이지네이션으로 계속 넘겨보기).
- 각 이미지: 기존 GT 전체 박스(회색) + 보정 GT 전체 박스(라벨별 색, 범례 참고)를 같이
  그려서, 안 바뀐 곳은 회색이 색 밑에 가려지고 바뀐 곳만 회색 테두리가 삐져나와 보인다.
- `train.jsonl`(기존)과 `train_refined.jsonl`(보정본) 둘 다 `combined/`에서 그대로 읽는다
  — 두 파일 모두 이 서버가 켜져 있는 동안 옮기거나 지우지 말 것.

## 모드 설명 (`--mode`)

| 모드 | 방식 | 장점 | 단점 |
|---|---|---|---|
| `union` | GT를 OCR extent로 감싸기만(항상 확장) | 텍스트 잘릴 위험 0 | 헐겁거나 어긋난(offset) GT는 못 고침 — 실측상 이런 케이스가 다수라 비추천 |
| `snap` | 이동량/IoU/면적비 캡으로 보수적 보정 | 안전 | 캡 때문에 많이 뒤틀린 GT는 오히려 거부됨 |
| `loose` (최종 채택) | 캡을 풀고 "GT 텍스트 재현율"로 신뢰도 판단 | 뒤틀린 GT도 안전하게 교정 | 로직이 가장 복잡 |

`loose` 모드 안전장치:
- **재현율 3단계**: OCR이 GT 텍스트 90%+ 잡으면 완전 스냅(큰 이동 허용) / 60~90%면 확장만
  (수축 금지) / 그 미만이면 변경 폭이 작을 때만 허용.
- **선두 마커 보호**: 문장 맨 앞이 원문자·불릿·괄호숫자·로마숫자·화살표·대시 등으로
  시작하면 그 경계(긴축의 시작점)는 안으로 밀지 않음.
- **교차축 보호**: 텍스트 안에 문자/숫자/공백이 아닌 문자가 하나라도 있으면(`*`, `'`, `-`,
  괄호, 마침표 등 사실상 대부분의 문장) 텍스트 흐름의 교차축은 축소 금지 — 가로쓰기
  문단이면 세로폭(높이), 세로쓰기 컬럼이면 가로폭(너비).
- **매칭은 IoU가 아니라 포함율**(OCR조각∩GT / OCR조각 면적) + y-band 게이트로 1:1 배정.

## 현재 산출물 위치

- `/home/jhyeo/finetuning/finetuning_dataset/combined/train.jsonl` — 원본(불변)
- `/home/jhyeo/finetuning/finetuning_dataset/combined/train_refined.jsonl` — 보정본
- `/home/jhyeo/finetuning/finetuning_dataset/combined/train_refined_viewer.html` — 정적 뷰어
- 라이브 서버: `http://localhost:8092` (이 폴더의 `app_refined.py` 실행 중이어야 함)
