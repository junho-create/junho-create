# 신규 배치 GT 자동 레이블링 파이프라인

원본 이미지 폴더 하나만 주면 `scripts/run_pipeline.sh`가 끝(`final_gt.jsonl`)까지 이어서
돈다. batch5400(5,400장)에서 처음부터 끝까지 실행하며 부딪힌 실패들을 전부 스크립트에
반영해 뒀다 — 이 문서는 "왜 이렇게 했는지"와 "막히면 뭘 봐야 하는지"를 남긴다.

## 실행

```bash
cp configs/batch5400.yaml configs/<새배치>.yaml   # paths.work_dir 만 새 경로로 수정
scripts/run_pipeline.sh configs/<새배치>.yaml <원본이미지_디렉터리> [max_uncovered_new]
```

- `max_uncovered_new` 생략 시 0.25 (아래 "임계값 재보정" 참고, 배치마다 값이 달라질 수 있음)
- 각 단계는 대부분 **resume-safe**다(이미 처리한 id는 건너뜀) — 중간에 죽어도 그냥 다시
  실행하면 이어서 간다. 단, 6단계(hardcase judge)에서 에러 난 레코드만 골라 재처리하고
  싶으면 "에러만 재처리" 절 참고.
- 소요 시간은 배치 크기와 GPU 상황에 따라 크게 다르다. 참고치(batch5400, Hard 3,483건
  기준): cmcv 전체 ~2시간, hardcase judge(workers=4) ~3~4시간.

## 단계 개요

| # | 단계 | 스크립트/명령 | GPU |
|---|---|---|---|
| 1 | 원본 인제스트 | `ocr_filter.cli io build-images` | 안 씀 |
| 2 | CMCV 3모델 기동 | `ocr_filter.cli models serve --only target external_a external_b` | 0,2,3 |
| 3 | CMCV 실행 | `ocr_filter.cli cmcv run` | 2에서 기동한 서버 |
| 4 | Hard 티어 rescue | `ocr_filter.cli rescue run` | 안 씀(CPU, N:M bbox 매칭) |
| 5 | CMCV 내리고 judge 기동 | `models stop` → `models serve --only judge` | 0,2,3 (PP=3) |
| 6 | hardcase judge | `ocr_filter.cli hardcase judge --workers 4` | 5에서 기동한 judge |
| 7 | 3단계 결과 머지 | `scripts/build_final_output.py` | 안 씀 |
| 8 | export→OCR 부여→필터 | `export gt-html` → `add_ocr` → `filter_e21grid.py` | add_ocr만 GPU 선택적 |
| 9 | 표 태그 paddle 교체 | `scripts/apply_table_paddle_refine.py` | 안 씀 |

### 왜 2/3모델과 judge가 순차인가
GPU 3장(H100 80GB)만 쓸 수 있고, target(GPU0)/dots.ocr(GPU2)/paddle(GPU3) 3모델이
그 3장을 전부 채운다. judge(122B, PP=3)도 GPU 0+2+3 전부 필요 — **동시 서빙 불가**,
반드시 CMCV를 끝내고 내린 뒤에 judge를 올려야 한다(`configs/models.yaml` 참고).

## 알아둬야 할 함정들 (전부 실측으로 확인된 것)

### 1. hardcase judge는 workers=32 쓰면 안 됨
render-then-verify(표 HTML → playwright Chromium 스크린샷)가 요청마다 Chromium을
새로 띄운다. workers를 높게 잡으면(32로 실측) 동시 인스턴스 경합으로
`Page.captureScreenshot: Unable to capture screenshot` 에러가 대량 발생해서 표 검증이
전부 스킵된 채 unresolved 처리된다. **workers=4**가 3,483건 전체를 에러 없이 완주한
값이다. 더 올리고 싶으면 Chromium 동시 인스턴스 한계부터 별도로 검증할 것.

이미 workers=32로 돌리다 에러가 많이 쌓인 상태에서 복구하려면(재실행은 안 하고
에러만 골라내고 싶을 때):
```python
import json
recs = [json.loads(l) for l in open("$WORK_DIR/hardcase_judge.jsonl")]
ok = [r for r in recs if r.get("resolved") or not r.get("error")]
with open("$WORK_DIR/hardcase_judge.jsonl", "w") as f:
    for r in ok:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
```
그 다음 `hardcase judge --workers 4`를 다시 실행하면 남은 것(에러+미처리)만 처리한다.

### 2. add_ocr.py는 정확한 인자와 실행 위치가 필요
- 실제 경로: `/home/jhyeo/finetuning/vlm/data/add_ocr.py` (다른 비슷한 경로들 —
  `tsr_eval/`, `tsr_test/` 등 — 은 다른 목적의 사본이니 헷갈리지 말 것)
- `python3 -m data.add_ocr ...` 형태로 **`/home/jhyeo/finetuning/vlm` 안에서** 실행해야
  내부 `from utils...` 임포트가 풀린다.
- 필수 인자: `--input --output --ocr_device --ocr_lang --bbox_scale`
- **GPU 경합 주의**: CMCV/judge 서버가 GPU를 90%+ 채운 상태에서 이 스텝을 GPU로 돌리면
  OOM 위험이 크다. 안전하게 가려면 `--ocr_device cpu` (2,262장 기준 CPU로 약 30분).
  judge를 이미 내린 뒤라 GPU가 비어 있다면 `--ocr_device gpu:0`로 훨씬 빠르게 가능.

### 3. filter_e21grid.py는 디렉터리+split 파일명 구조를 요구
`--data_dir`/`--out_dir`를 받고, 내부에서 `{data_dir}/{split}.jsonl`을 찾는다
(`--splits` 기본값 `train valid test`). 단일 파일을 넘기면
`unrecognized arguments` 에러가 난다 — 파이프라인 스크립트가 이미 `train.jsonl`
하나짜리 임시 디렉터리로 감싸서 호출한다.

### 4. `uncovered_ocr` 임계값(0.05)은 이 새 파이프라인에 안 맞음 — 재보정 필요
`filter_e21grid.py --max_uncovered_new`의 기본값 0.05는 **다른(구형) 라벨링 파이프라인**
으로 캘리브레이션된 값이다. 이 파이프라인은 Medium 라벨을 dots.ocr 출력을 그대로
pseudo-label로 채택하는데, dots.ocr 자체가 PaddleOCR(범용 OCR, add_ocr.py가 별도로
돌리는 것)만큼 촘촘하게 페이지를 segment하지 않는다 — **순수 dots.ocr 라벨조차
uncovered_ocr 중앙값이 0.15~0.18대**로 나온다(batch5400 실측). 기본값 그대로 돌리면
2,262건 중 79%가 날아간다.

**새 배치를 돌릴 때마다 반드시 분포를 먼저 확인할 것.** 대략적인 확인 방법:
```bash
# with_ocr.jsonl 생성 후, filter_e21grid.py를 --dry_run 없이 여러 임계값으로 시험 실행하거나
# 직접 uncovered 분포를 계산 (scripts/apply_table_paddle_refine.py의 raster()/iou() 코드가
# 참고가 됨 — filter_e21grid.py 자체의 raster()/SCALE=1000 로직과 동일한 방식).
```
목표 통과 건수(전체의 ~30%대)에 맞춰 임계값을 역산하는 게 실전적이었다
(batch5400: 0.05→471건, 0.25→1,586건 중 실제로는 short_ocr/invented_ellipsis까지 겹쳐
최종 1,571건).

### 5. 표(Table) 태그는 기본적으로 dots.ocr 것이 들어간다 — paddle로 바꾸려면 9단계 필수
`cmcv/run.py`의 Medium pseudo-label과 `build_final_output.py`의 text_easy_recovered
경로 둘 다 `external_a`(dots.ocr) 전체를 그대로 채택한다(표 포함). `rescue.py`만
표에 한해 paddle을 채택하고(`_ADOPT_FROM = {"Table": "b"}`), hardcase judge(122B)는
둘 다 참고해서 새로 생성한다 — 즉 **표 태그 출처가 소스마다 다르다.**

9단계(`apply_table_paddle_refine.py`)가 `combined_e24_table_refined/CHANGES.md`의
검증된 규칙(bbox IoU 매칭 + 글자수비율/앞뒤15자/전체 유사도 게이트, paddle 셀 개행은
공백 치환)을 그대로 이식해 **가능한 곳만** paddle로 교체한다. 게이트를 통과 못 하면
원본(dots.ocr 또는 judge 생성분)을 그대로 유지 — 무리하게 교체해서 내용이 달라지는
사고를 막기 위한 안전장치다(batch5400 실측: 표 1,633개 중 881개=54% 교체, 나머지는
매칭 실패/게이트 실패로 원본 유지).

### 6. GPU에 안 보이는 잔여 점유(ghost memory)
`nvidia-smi`가 "No running processes found"라고 해도 실제로는 몇 GB가 점유된 채로
안 풀리는 경우가 있었다(다른 컨테이너/세션 소유로 추정, 우리가 못 내림). judge 서버
기본 `--gpu-memory-utilization`(~0.9)이 이 여유분을 안 남겨서 실제 이미지 요청(멀티모달
인코더 캐시) 중 NCCL OOM으로 통째로 죽은 적이 있다. `configs/models.yaml`의 judge
블록이 이미 0.85로 낮춰서 이 문제를 우회해 뒀다 — 다른 모델도 같은 증상이면 우선
`--gpu-memory-utilization`을 낮춰볼 것.

### 7. vLLM 크래시 후 워커 프로세스가 안 죽고 GPU를 계속 물고 있을 수 있음
APIServer 프로세스가 fatal error로 죽어도 `VLLM::Worker_PP0/1/2` 서브프로세스가 살아서
GPU 메모리를 계속 점유하는 경우가 있었다(로그는 "Parent process exited, terminating
worker"라고 주장해도 실제로는 안 죽음). `nvidia-smi`엔 프로세스가 안 보일 수 있으니
`ps aux | grep Worker_PP`로 찾아서 직접 `kill -KILL` 해야 GPU가 풀린다.

## 재실행/부분 재실행

- 어느 단계든 **같은 out 경로로 다시 실행하면 이미 처리한 id는 건너뛴다** (io 제외 —
  unified.jsonl은 통째로 다시 씀, 필요하면 `--out`을 다르게 주고 나중에 합칠 것).
- 6단계(hardcase judge)만 에러 재처리가 필요할 수 있음 — "함정 1" 참고.
- 9단계(표 paddle 교체)는 8단계 산출물(`final_gt.jsonl`)에 대해서만 다시 돌리면 되고,
  cmcv_results.jsonl/unified.jsonl은 그대로 재사용 가능(모델 재호출 없음, 몇 초~분 단위).

## 산출물 (work_dir 기준)

```
unified.jsonl              # 1) 원본 인제스트
cmcv_results.jsonl         # 3) CMCV: E/M/H 티어 + pseudo-label
rescue_results.jsonl       # 4) Hard→Medium-rescued
hardcase_judge.jsonl       # 6) Hard 티어 judge 결과 (resolved=True 만 자동확정)
final_dataset.jsonl        # 7) 머지 (id/tier/label/label_source)
gt_html.jsonl              # 8) SFT 학습 포맷 (gt_html 필드, ocr_info 아직 빈 배열)
with_ocr.jsonl             # 8) + ocr_info 채워짐 (filter_e21grid 입력)
final_gt_before_tablefix.jsonl  # 8) 필터 통과, 표는 아직 원본(dots/judge)
final_gt.jsonl             # 9) 최종 산출물 — 표 가능한 곳만 paddle로 교체됨
```
