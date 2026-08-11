# combined_e24_refined — layout 페이지 Table GT 감사

`combined_e24_refined` 의 **layout 레코드**(`prompt_style == "unified_layout"`) 안에 들어
있는 `<div data-label="Table">` 의 GT HTML 품질을 LLM-judge 로 점검하고, 의심스러운
페이지를 사람 검수 큐로 뽑아낸다. TSR crop(`unified_table_with_ocr`)은 대상이 아니다.

**고치지 않는다. 걸러내기만 한다.** 원본 jsonl 은 읽기만 하고 절대 수정하지 않는다.

## 대상

| split | Table 포함 layout 페이지 | Table div |
|---|---|---|
| train | 4,101 | 6,248 |
| valid | 473 | 693 |
| test | 456 | 712 |
| **합계** | **5,030** | **7,653** |

페이지당 표: 1개 69.9% / 2개 18.3% / 3개 이상 11.8% (최대 27개).
Table 라벨만 있는 layout 페이지 38장도 포함.

## 방식

페이지 1장 = judge 호출 1회. 이미지 2장을 넣는다.

- **A** 원본 페이지 이미지 + Table bbox 를 빨간 박스와 `#1`, `#2` … 번호로 오버레이
- **B** 그 페이지 Table div 의 GT HTML 만 같은 번호를 붙여 브라우저로 렌더한 스크린샷

judge 는 번호가 같은 표끼리만 비교해 표별로 6개 지표(범위/기본구조/병합구조/셀대응/
텍스트/시각)를 1~5 로 매기고, 페이지에 있는데 라벨이 없는 표도 보고한다.

**재시도 없음** — 호출·파싱이 한 번에 안 되면 그 페이지는 `JUDGE_ERROR` 로 기록하고
바로 넘어간다. 이 건들도 사람이 봐야 하므로 검수 큐에 그대로 들어간다.

judge: GenOS serving **776** (`Qwen3.5-397B-A17B-FP8-Instruct`, 사내 서빙이라 과금 없음).
키는 `/home/jhyeo/finetuning/eval_318/tsr_test_eval/.env` 의 `genos_key`.

> 776 은 thinking 모델이다. `enable_thinking: false` 를 안 넘기거나 `max_tokens` 가
> 작으면 `reasoning_content` 가 예산을 다 먹고 `choices` 가 빈 채로 돌아온다.
> `judge.py:JUDGE_CONFIG` 에 이미 반영돼 있다.

## 실행

```bash
cd /home/jhyeo/finetuning/finetuning_dataset/gt_table_audit
export PYTHONPATH=/home/jhyeo/ocr_file_filter/labeler

python3 extract.py                                  # manifest.jsonl (약 45초)
python3 run_audit.py --preflight --out results/preflight.jsonl   # 1건만 확인
python3 sanity_corrupt.py --n 8                     # 판별력 테스트 (아래 참고)
python3 run_audit.py --sample 100 --workers 8       # 파일럿
python3 report.py                                   # flagged / summary / review.html
python3 run_audit.py --workers 8                    # 전량 (audit.jsonl 이어서 씀)
python3 run_audit.py --retry-errors --workers 8     # JUDGE_ERROR 건만 재실행
```

`run_audit.py` 는 `results/audit.jsonl` 에 이미 있는 key 를 건너뛰므로 중간에 끊겨도
같은 명령으로 이어서 돌리면 된다.

## 판정 기준

| status | 조건 |
|---|---|
| `FLAG` | 정적 플래그 있음 / 어떤 표든 지표 하나라도 `--flag-threshold`(기본 4) 이하 / 미라벨 표 발견 |
| `JUDGE_ERROR` | 1회 호출에서 API 오류·타임아웃·빈 응답·JSON 파싱 실패·표 개수 불일치 |
| `RENDER_ERROR` | Playwright 렌더 실패 |
| `PASS` | 그 외 (모든 표의 모든 지표 5점) |

기본 임계값 4 는 "조금이라도 이상하면 뽑는다" 쪽이다. 너무 많이 걸리면
`--flag-threshold 3` 으로 낮춘다.

### 정적 플래그 (judge 없이도 확실한 것)

`extract.py` 가 붙인다. 실측:

- `no_table_tag` **120개 표** — Table div 안에 `<table>` 이 아예 없음.
  실제로 열어 보면 표가 아니라 본문 블록·검색 폼·주석 문단을 Table 로 라벨한 것들이다.
- `tiny_bbox` **120개 표** — bbox 면적이 페이지의 1% 미만 (p5=1.8%, 중앙값=14.9%)
- `empty_table` **7개 표** — div 내부가 빈 문자열

## 산출물

```
manifest.jsonl          대상 5,030페이지 + 표별 bbox/HTML + 정적 플래그
results/audit.jsonl     페이지 1건 = 1행 (status, 표별 6지표 점수, 사유)
results/flagged.jsonl   검수 대상만, 심각한 것부터 정렬
results/flagged_keys.txt  key 목록 (--keys-file 로 재실행할 때)
results/summary.json    split 별 집계, 지표 평균, error_type 빈도
results/review.html     오버레이/렌더 나란히 보는 검수 페이지
work/overlays/*.jpg     A 이미지
work/renders/*.png      B 이미지 (+ 같은 이름의 .html)
```

`review.html` 은 이미지를 **상대 경로로 참조**한다 (base64 로 박으면 수 GB). 다른
머신에서 열려면 `python3 report.py --embed --max 200`.

## 파일

| 파일 | 역할 |
|---|---|
| `extract.py` | jsonl → `manifest.jsonl`, BeautifulSoup 로 Table div 파싱 + 정적 검사 |
| `render.py` | 오버레이(PIL) / GT 표 렌더(Playwright). 단독 실행하면 앞 N건만 만들어 본다 |
| `judge_prompt.md` | 페이지 단위 judge 프롬프트 + 출력 스키마 (`gt_table_audit_v1`) |
| `judge.py` | GenOS 776 클라이언트(재시도 없음) + 응답 파싱/검증 |
| `run_audit.py` | 스레드 풀 오케스트레이션, 재개, 상태 판정 |
| `sanity_corrupt.py` | 판별력 테스트 — GT 를 인위로 훼손해 judge 가 잡는지 확인 |
| `report.py` | flagged / summary / review.html 생성 |

## 재사용한 기존 코드

- `genos.doc_parser.labeler.core.renderer.Renderer` — Playwright, 스레드 안전
- `genos.doc_parser.labeler.core.vlm_client.VLMClient` — `judge.SingleShotVLMClient`
  가 상속. `call` 만 오버라이드해 재시도를 없앴다 (원본은 3회 하드코딩)
- `labeler/tsr/templates/compare_template.html` 의 표 CSS
- `labeler/tsr/prompts/evaluate_prompt.md` 의 6지표 정의·점수기준·enum
- `labeler/core/pipeline.py` 의 thread-local Renderer 패턴

## 왜 판별력 테스트를 먼저 돌리나

`ocr_filter/hardcase/prompts.py:68-77` 주석에, 지표를 잘게 쪼갠 judge 가 판별력을 잃어
"박스를 20% 밀고 절반을 지운" 합성 결함조차 5점/PASS 로 통과시킨 사례가 남아 있다.
`sanity_corrupt.py` 는 멀쩡한 GT 에 행 삭제 / 숫자 자리바꿈 / span 값 변경을 넣고
judge 가 실제로 감점하는지 본다. 여기서 못 잡으면 전량 실행은 의미가 없다.
