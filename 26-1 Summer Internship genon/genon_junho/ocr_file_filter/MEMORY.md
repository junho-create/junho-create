# MEMORY — ocr_file_filter 이어받기 문서

> 다른 서버에서 clone 해 이어 작업하는 클로드를 위한 핸드오프. **먼저 이 문서를 읽고 시작.**
> 마지막 갱신: 2026-07-13 (이 B200 서버에서 모델 브링업 + io 실데이터 검증 완료)

## 이 레포가 하는 일
VL 문서 파싱 데이터셋 **큐레이션/필터 엔진** (MinerU2.5-Pro 데이터 엔진 방식).
학습 스크립트가 아니라, "어떤 샘플을 어떤 라벨로 학습에 넣을지"를 결정하는 3축 엔진.
출력물 = Easy/Medium 자동 데이터셋 + Medium pseudo-label + Hard 인간주석 큐.
qwen3vl-8b 학습 자체는 이 레포 밖의 일. 어느 서버든 clone 후 config만 고치면 동작하도록 설계.

## 확정된 설계 결정 (locked)
- **타겟 모델(CMCV 기준)**: base **Qwen3.5-9B** + LoRA 어댑터 (병합된 체크포인트 아님!).
  base: `_models/Qwen3.5-9B` (jhshin/shkim 원본 `/NHNHOME/.../shkim/models/Qwen3.5-9B` 복사, 19GB).
  LoRA: `_models/chandra_all_e18align_20260623_lora` (jhshin `.../output/chandra_all_e18align_20260623/student_sft/final` 복사, r=128/alpha=256, 931MB).
  vLLM `--enable-lora --lora-modules`로 서빙 (`ocr_filter/models/manifest.py` 의 `lora_path`/`lora_rank`).
- **외부 2종(CMCV 교차검증)**: `dots.ocr`(`rednote-hilab/dots.ocr`, HF 에서 직접 다운로드 — repo_id 확인 완료),
  `PaddleOCR-VL 1.6` (`jhyeo/paddleOCRVL1.6` 리포에 이미 준비된 모델/venv_vllm 을 그대로 위임 서빙,
  `serve_script` 필드로 연결). **셋 다 이 B200 서버에서 직접 서빙** — 원격/VPN 아님
  (이 서버는 huggingface.co 에 직접 접속 가능, 200 OK 확인됨).
- **GPU 배정**: target=0, dots.ocr=2, paddle=1 (mineru_work 가 쓰던 0/1 은 2026-07-13 에 정지시켜 확보).
- **vLLM 바이너리**: 이 리포 자체 venv 없음 — `jhyeo/paddleOCRVL1.6/venv_vllm/bin/vllm`(0.24, Qwen3_5ForConditionalGeneration
  지원 확인됨)을 `vllm_bin` 필드로 재사용. `jhyeo/mineru_work/venv`의 vllm(0.21)은 Qwen3.5 아키텍처 미지원이라 못 씀.
- **알려진 서버 함정**: `/tmp` 가 `noexec` 마운트라 triton/torchinductor 컴파일 캐시가 mmap 실행 실패
  (`failed to map segment from shared object`). `ocr_filter/models/serve.py`의 `_compile_cache_env()`가
  `TRITON_CACHE_DIR`/`TORCHINDUCTOR_CACHE_DIR`을 `~/.cache` 밑으로 돌려서 해결 — 이 서버에서 뭘 띄우든 필요.
- **모델 서빙 방식**: vLLM OpenAI-호환 엔드포인트 (포트 8001/8002/8080 — 8003 아님, paddle 쪽 자체 스크립트 기본값 8080 그대로 씀)
- **주 태스크**: 전체 페이지 파싱 (full-page)
- **렌더 실패 처리**: 선택지 B (문법오류 자동보정 별도 트랙 → 실패분만 Hard 큐)
- **`[0] io`는 raw 3소스 개별 파서로 구현** (tsr_test의 `build_unified_dataset.py` 재사용 안 함 — 사용자가
  명시적으로 개별 파서 방식을 선택함). `configs/default.yaml`의 `layout_src`/`table_src`/`images` 경로로
  raw JSONL을 직접 읽어 `{id, image_path, gt, source_type, meta}`로 변환.

## 원본 데이터 위치 (NHN 서버)
- 최종 학습 데이터: `.../_train_data/chandra_table_layout_divhtml_16886`
- 최종 학습 이미지: `.../chandra_table_layout_divhtml_16886/images`
- 레이아웃 원본 JSONL: `.../_train_data/layout_src_9984/labeler_converter_layout_source_9984.jsonl`
- 테이블 원본/분할: `.../_train_data/table_src_6902`
- (공통 프리픽스: `/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/`)
- 스케일 주의: 논문의 65.5M/3.9M/192K는 참고용. **실제 데이터는 ~17K 규모 → 정밀 큐레이션 1패스**.

## 전체 파이프라인 (실행 순서 — 의존성 때문에 인터리브됨)
```
[0] 원본 로드/통일   3소스 → 통일 스키마 JSONL {id, image_path, gt?, source_type, meta}
[1] Embed+Cluster    ViT 임베딩 → K-Means (페이지 수준 → 요소 수준 2단계)
[2] Taster CMCV      클러스터별 소량만 3모델 추론 → E/M/H 비율로 클러스터 난이도 추정
[3] DDAS 예산배분     S_i 계산 → N_i 할당 → 클러스터별 샘플링           ← ✅ 구현됨
[4] Full CMCV        샘플링셋 전체 3모델 교차검증 → 최종 E/M/H + pseudo-label
[5] 라우팅           Easy/Medium → 자동 데이터셋 | Hard → [6]
[6] Hard 정제        Render-then-Verify(Judge-Refine) → 문법오류 별도트랙 → 잔여만 사람 큐
```
CMCV는 두 번 호출: [2] 클러스터 난이도 추정용(taster만, 싸게) + [4] 최종 라벨링용(샘플링셋만).

## 구현 현황
| 스테이지 | 상태 | 위치 |
|---|---|---|
| 모델 브링업 | ✅ download/serve/status/stop, 이 서버에서 target/dots.ocr/paddle 3개 실기동 검증 완료 | `ocr_filter/models/`, `ocr_filter/cli.py` |
| DDAS | ✅ score(S_i)/allocate(N_i)/sample + 테스트 8/8 통과 | `ocr_filter/ddas/`, `tests/test_ddas.py` |
| io (0) | ✅ layout/table 개별 파서 + 실데이터 검증(16886건, 이미지 누락 0) + 테스트 3/3 통과 | `ocr_filter/io/`, `tests/test_io.py` |
| metrics | ✅ edit_distance(순수 Levenshtein) + TEDS/TEDS-S(직접 짠 Zhang-Shasha 트리편집거리, 외부 의존성 없음)
구현, 실데이터로 검증(자기비교=1.0). **의도적으로 교체 가능하게 설계** — `SCORERS` 딕셔너리 시그니처
`(gt, pred) -> float\|None` 만 지키면 구현 통째로 바꿔도 cmcv 쪽 안 건드림. CDM(수식)은 아직 미구현
placeholder(`None` 반환, 가짜 점수 안 냄) — dp-bench evaluate.py는 이 서버에 실존하지 않음(README에
설명된 gitlab 브랜치이고 clone 안 돼있음, [[tsr-test-data-pipeline-reuse]] 메모 정정 필요) | `ocr_filter/metrics/`, `tests/test_metrics.py` |
| normalize | ✅ `ocr_filter/cmcv/normalize.py` 에 통합됨 (3모델 출력 → `[{"category","text","bbox"}]`) |
| cmcv (2·4) | ✅ 구현 + 실서버 검증. **전체(16886건) 실행은 시작 직후 사용자가 중단시킴**(7~8h는 너무 길다고
판단) — 지금은 데모용 373건(레이아웃)만 `_work/cmcv_results.jsonl`에 있음 | `ocr_filter/cmcv/`, `tests/test_cmcv_normalize.py` |
| cluster (1) | ✅ ViT(`google/vit-base-patch16-224-in21k`) 임베딩 + 페이지→요소 2단계 KMeans, 합성데이터
테스트 4/4 + 실데이터 200페이지 스모크(8 페이지클러스터/43 요소클러스터) 검증 완료. **별도 venv 필요**
(`ocr_filter/venv`, torch/scikit-learn 은 `--system-site-packages` 로 재사용 + transformers 추가 설치—
시스템 python 은 PEP668 externally-managed 라 직접 설치 불가) | `ocr_filter/cluster/`, `tests/test_cluster_kmeans.py` |
| report | ✅ cmcv 결과를 Easy/Medium/Hard 티어별로 GT\|target\|dots.ocr\|paddle 4열 bbox 오버레이
HTML 갤러리로 생성 (`ocr_filter/report/`). Artifact 로 게시해서 봄 — JPEG q82 + max_side 1000 으로
용량 줄임(24MB→9MB, 15건 기준) |
| taster (2·3 연결) | ✅ cluster→소량CMCV→DDAS 연결, 스모크테스트 검증 완료 | `ocr_filter/taster/`, `tests/test_taster.py` |
| hardcase (6) | ✅ 로직 구현+mock 테스트 9/9 통과. judge 모델(NVFP4 251GB) 다운로드 중, 실서버 검증 아직 | `ocr_filter/hardcase/`, `tests/test_hardcase.py` |

## ⚠️ 2026-07-13 세션 후반 방향 전환 — 지금 데이터는 데모용, 실데이터는 나중에 새로 수집
사용자가 명시: **지금 쓰는 16886건(jhshin 원본)은 파이프라인 동작 검증용 데모일 뿐**, 실제로는
나중에 새 데이터를 수집해서 다시 돌릴 예정. 그래서:
- cmcv 전체(16886건, ~7.5~8h) 를 지금 다 돌릴 필요 없음 — 이미 중단시킴. 작은 샘플(수백 건)로
  파이프라인이 맞게 동작하는지만 확인하는 게 지금 목적.
- **[1] cluster 를 나중에 붙여야 하는 이유가 바로 이 7~8시간 문제였음**: 원래 설계는
  cluster → Taster CMCV(클러스터별 소량만) → DDAS → Full CMCV(샘플링된 것만) 순서라, 전체
  16886건에 CMCV 를 무조건 다 돌릴 필요가 없다. 지금은 cluster 없이 곧장 전체 CMCV 를 시도했다가
  시간 문제로 막힌 것 — **다음엔 cluster 결과로 taster 샘플을 줄여서 CMCV 돌리는 흐름으로 정리할 것.**
- 데이터가 바뀔 걸 아니까, cluster/cmcv/report 전부 `--limit`/`--per-tier` 로 작은 규모부터
  돌려보고 검증하는 습관 유지. 하드코딩된 경로(jhshin 원본)도 나중에 새 데이터 경로로 그냥
  `configs/default.yaml` 만 바꾸면 되게 이미 설계돼 있음.

## ⚠️ CMCV 관련 중요 발견 (2026-07-13)
- **PaddleOCR-VL-1.6-0.9B 는 target/dots.ocr 처럼 프롬프트로 부르면 안 됨.** 원래
  `PaddleOCRVL` 파이썬 파이프라인(레이아웃검출 PP-DocLayoutV3 + 크롭별 인식) 안에서만 쓰도록 학습된
  "인식 전용" 컴포넌트라, Chandra 프롬프트든 평범한 지시문이든 텍스트를 주면 반복루프/환각으로
  무너짐. **텍스트 프롬프트 없이 이미지만 줘야** 정상적인 OCR 텍스트를 뱉는다
  (`ocr_filter/cmcv/prompts.py`의 `external_b` 분기, `client.py`가 content=None 이면 텍스트 파트 생략).
  구조화된 카테고리 없이 순수 인식 텍스트 한 덩어리만 나오므로 `normalize.parse_paddle_output`
  이 `[{"category":"Text","text":raw}]` 로 감싼다.
- 이 때문에 gt/모델간 채점은 **`full_text()`(태그 벗긴 평문) + edit_distance 로 통일**해서 함
  (TEDS는 target/dots.ocr 둘 다 구조화된 Table 을 낼 때만 의미 있어서 `teds_score` 필드에
  보너스 신호로만 남김, 주 지표(`gt_score`)로는 안 씀 — paddle 이 항상 낮게 나와버리기 때문).
- **처리 속도**: GPU0(target, 9B)이 병목 — worker=32 에서 레코드당 ~1.64초. 16886건 전체 ~7.5~8시간
  추정. `_work/cmcv_run.log`/`_work/cmcv_results.jsonl` 로 진행상황 확인 (이미 처리한 id는
  재실행시 자동 스킵, resumable).

## ✅ taster (cluster → 소량 CMCV → DDAS) 연결 완료 (2026-07-13)
`ocr_filter/taster/` 신설: `sample_taster_ids`(클러스터당 `taster_per_cluster`개 무작위 샘플) →
`run_cmcv(..., ids=...)`(cmcv/run.py 에 `ids` 필터 추가함) → `aggregate_tier_counts`(클러스터별
E/M/H 집계, allocate_budget 이 요구하는 키 집합과 정확히 맞추려고 누락 클러스터는 0/0/0 으로 채움)
→ `ocr_filter.ddas.score_clusters`/`allocate_budget`(기존 구현 그대로 재사용). CLI:
`python -m ocr_filter.cli taster run --n-per-cluster 8`. 500페이지/64클러스터 데이터로 실서버
스모크테스트 완료(113 taster 샘플, 에러 0, DDAS 배분까지 정상). 테스트 5/5 (`tests/test_taster.py`).
**주의**: `--ddas-n-total` 기본값이 전체 크기라서 기본으로 돌리면 전부 다 뽑힘(자름 없음) —
실제 "고가치 클러스터에 집중배분" 효과를 보려면 이 값을 전체보다 작게 줘야 함.

## 2026-07-13 후반: 전체 데이터(9984 레이아웃 페이지) 클러스터링 + taster 진행
사용자가 "14000개(=전체 데모 데이터)에 대해 clustering부터 E/M/H 분류까지" 요청. 500페이지
샘플 기준 3분53초 걸려서, 9984페이지 전체는 ~80분 추정 (CMCV 7~8시간보다 훨씬 빠름 — 이게
cluster→taster 를 먼저 만든 이유였음, 전체 리허설로 검증 중).
`./venv/bin/python3 -m ocr_filter.cli cluster build --out _work/clusters.jsonl` 백그라운드 실행 중
(로그: `_work/cluster_run.log`). 끝나면 `taster run` 으로 이어서 전체 데이터 기준 클러스터 난이도 산출.

## ✅ [6] hardcase 구현 완료 (2026-07-13)
`ocr_filter/hardcase/` — Judge-and-Refine(1단계) + Targeted Expert Annotation 사전라벨(2단계).
- **1단계**: target 예측(bbox 오버레이 렌더링, `ocr_filter.report.render.draw_boxes` 재사용) vs
  원본을 heavy judge VLM 에 보여줘서 판정(JSON: pass/issues) → FAIL 이면 원본+이슈만 다시 줘서
  재교정(refine) → 최대 `max_refine_rounds`(기본 3) 반복. `ocr_filter/hardcase/pipeline.py`.
- **2단계**: 1단계에서 안 풀린(unresolved) 것만, **judge 와 다른 계열이어야 하는 "독립" 모델**
  (`configs.hardcase.prelabel_vlm`)이 원본만 보고 초안을 새로 만듦 — 데이터 누출(같은 모델
  계열이 같은 맹점을 공유) 방지 목적. 지금은 prelabel_vlm 도 임시로 judge_vlm 과 같은 엔드포인트를
  가리키지만, **나중에 OpenRouter 키 생기면 `prelabel_vlm` 블록만 `provider: openrouter` 로 바꾸면
  코드 수정 없이 Gemini 등으로 스왑됨** (`ocr_filter/hardcase/client.py` 가 범용 OpenAI-호환 호출기).
- CLI 로 **1/2단계 분리 실행 가능**: `hardcase judge`(1단계만, `_work/hardcase_judge.jsonl`) /
  `hardcase prelabel`(2단계만, 1단계 결과 읽음, `_work/hardcase_prelabel.jsonl`) / `hardcase run`(둘다).
- **3단계 (Labeler 연동)**: `hardcase export` 로 2단계 결과를 `labeler` 의 Layout UI 와 연동.
  `_work/hardcase_prelabel.jsonl` 의 초안 bbox를 읽어 `labeler/` 파이프라인의 **평가(evaluator) 및 리뷰 UI 생성** 단계를 그대로 재활용함. VLM 평가는 생략되지 않으며(judge 모델을 evaluator로 재사용), `_work/labeler_export/review.html` 이 생성되어 Monaco 에디터로 사람이 직접 수정할 수 있음. (`ocr_filter/hardcase/export_labeler.py` 가 어댑터 역할)
- 테스트 9개: judge/refine/prelabel 전부 `call_fn` 의존성 주입으로 mock — 실제 vLLM 서버 없이도
  로직(판정 파싱, 재시도 루프, 파싱실패시 퇴행방지) 검증됨.

**judge 모델**: `nvidia/Qwen3.5-397B-A17B-NVFP4`(251GB, B200 네이티브 FP4) 선택 — 원본
`Qwen/Qwen3.5-397B-A17B`(bf16 807GB)는 TP=8 필요해서 지금 떠있는 target/dots.ocr/paddle을 다
내려야 하는데, NVFP4 는 TP=4 로 비어있는 GPU 3~6만 쓰면 돼서 기존 서버 안 건드림. 2026-07-13
다운로드 완료 후 서빙까지 끝남 (`_models/Qwen3.5-397B-A17B-NVFP4`, `configs/models.yaml` 의
`judge` 항목, GPU 3~6, `models status` UP 확인). 아래 "judge 모델 실서버 서빙 + hardcase 실검증
완료" 절 참고.

## ✅ judge 모델 실서버 서빙 + hardcase 실검증 완료 (2026-07-13)
- `configs/models.yaml` 에 `judge` 항목 추가(gpus="3,4,5,6", tensor-parallel 자동 4, venv_vllm
  재사용) → `models serve --only judge` 로 정상 기동 확인, `models status` UP.
- **중요 버그+수정**: `Qwen3.5-397B-A17B-NVFP4` 는 **thinking(추론) 모델**이라 답 내기 전에
  수천~2만자 분량 사고과정을 씀. `ocr_filter/hardcase/client.py` 의 기존 `max_tokens=4096` 으로는
  사고과정 도중 잘려서 JSON 을 한 번도 못 내고 매번 `parse_error=True` 가 났었음 —
  **`max_tokens` 를 16384 로 올려서 해결**(응답 안에 `</think>` 태그로 사고과정과 최종 JSON 이
  구분됨, `parse_judge_verdict` 는 마지막 `{...}` 만 뽑으므로 그대로 잘 작동). 콜당 40~130초
  정도 걸림(사고 분량에 따라 편차 큼). 실제 3건 테스트: 1건은 라운드를 거치며 실제로 정확히
  개선되어 resolved=True 까지 감(중복 bbox 6개 정확히 짚어내고 refine 후 제거됨 확인) — 판정
  품질 자체는 매우 좋음. 나머지 2건은 여전히 가끔 parse_error 남음(복잡한 페이지일수록 사고
  분량이 늘어 16384 로도 가끔 부족) — **더 늘리거나(예: 24000+, 느려짐), 아니면
  `chat_template.jinja` 에 있는 `enable_thinking` 옵션을 요청 시 false 로 꺼서 사고 생략하고
  바로 답하게 하는 것도 고려해볼 만함(속도/안정성 ↑, 판정 깊이는 약간 ↓ 할 수 있음)** — 아직
  안 해봄, 다음에 필요하면 `hardcase/client.py` 의 `call_vlm` 에 `extra_body: {"chat_template_kwargs":
  {"enable_thinking": false}}` 추가하는 식으로 넣으면 됨(vLLM 이 Qwen3 계열에 대해 지원).

## ✅ 전체 데이터(9984페이지) 클러스터링 완료 (2026-07-13)
`_work/clusters.jsonl`: 9984페이지 → 64개 페이지클러스터(크기 16~406, median 141) + 129787개
요소 → 2021개 요소클러스터. `taster run` 을 전체 데이터로 실행 시작함(`_work/taster_run.log`,
`_work/taster_cmcv_results.jsonl`) — 완료되면 클러스터별 E/M/H 분포와 DDAS 배분 결과 확인.

## 다음에 할 일 (권장 순서)
1. **`_work/taster_run.log` 완료 확인 후 결과 리뷰** — 클러스터별 S_i/배분 결과가 합리적인지,
   특정 클러스터가 유독 Hard 비율이 높은지 등 확인.
2. **judge parse_error 잔여 튜닝** — max_tokens 추가 인상 또는 enable_thinking=false 실험.
3. **실데이터 교체 시**: `configs/default.yaml` 의 `paths:` 만 새 데이터 경로로 바꾸면 나머지는
   그대로 재사용 가능하도록 설계돼 있음 (io/cmcv/cluster/taster/hardcase/report 전부 경로 하드코딩 없음).

## 이미 만든 것 실행법
```bash
# 모델: 다운로드 → 서빙 → 상태 (target/dots.ocr/paddle 전부 이 서버에서 직접 서빙)
python -m ocr_filter.cli models download --only external_a   # dots.ocr 만 (target/paddle 은 로컬에 이미 있음)
python -m ocr_filter.cli models serve --dry-run               # 명령 미리보기
python -m ocr_filter.cli models serve                          # 3개 다 기동 (또는 --only target external_b 등)
python -m ocr_filter.cli models status                         # UP/DOWN 헬스체크
python -m ocr_filter.cli models stop                            # 종료

# io: 원본 3소스 → 통일 스키마 JSONL
python -m ocr_filter.cli io build                               # ./_work/unified.jsonl (16886건)

# cmcv: 3모델 채점 (resumable, --limit 으로 일부만)
python -m ocr_filter.cli cmcv run --limit 300 --workers 32       # 시험용 소량 — 전체(16886)는 ~7.5~8h!
python -m ocr_filter.cli cmcv run --limit 300 --source-type table

# cluster: ViT 임베딩 + 2단계 KMeans — **반드시 venv/bin/python3 로 실행** (transformers 는 이
# venv 에만 설치돼 있음, 시스템 python 은 PEP668 로 pip install 자체가 막혀 있음)
./venv/bin/python3 -m ocr_filter.cli cluster build --limit 500

# report: cmcv 결과 티어별 GT|target|dots.ocr|paddle bbox 갤러리 HTML
python -m ocr_filter.cli report gallery --per-tier 5 --out gallery.html

# 테스트/데모
python3 -m pytest tests/ -q                                      # 전체 28개
python3 tests/test_ddas.py                                       # DDAS 합성 데이터 배분 시연
python3 tests/test_io.py                                         # io 합성 fixture 시연
python3 tests/test_metrics.py                                    # 실데이터 gt_html 자기비교 TEDS=1.0 시연
```
DDAS 핵심: `N_i = min( (S_i+α)^β / Σ(S_j+α)^β · N_total , |C_i| )`, 캡에 걸린 잉여 예산은
water-filling 으로 재분배 → `sum(N_i) == min(N_total, Σ|C_i|)` 보장 (`allocate.py`).

io 핵심: raw 소스의 `image_path`는 더 이상 그 경로에 존재하지 않음(build_unified_dataset 단계에서
이미 `images/{layout,table}/<basename>` 으로 평탄화·복사돼 있음) — 그래서 파서가 `Path(raw_image_path).name`
으로 basename만 뽑아 `images_root/{layout,table}/<basename>` 로 재매핑한다 (`ocr_filter/io/layout.py`,
`ocr_filter/io/table.py`). table_src 원본엔 `id` 필드가 없어 `table:<split>:<stem>` 으로 합성.
