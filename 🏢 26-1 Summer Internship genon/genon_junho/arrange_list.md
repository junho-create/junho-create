# 정리 후보 리스트 (2026-07-27 기준)

원칙:
- 여기 있는 건 전부 "확인 후 직접 tmux에서 지우기" 용 리스트입니다. 자동으로 지운 건 없습니다.
- **0701 이전에 생성/수정된 파일·폴더는 전부 제외**했습니다 (건드리지 않기로 한 것들).
- 워크플로우 3개(데이터 필터링 / 파인튜닝 / 평가) 기준으로 나눴고, 마지막에 "실행 위치가 흩어진 중복"을 따로 모았습니다.
- 각 항목: 경로 / 크기 / 마지막 수정일 / 상태.

```
안전도 표시
[삭제 안전]   : venv, 캐시, 로그, 이름부터 trash인 것 — 재생성 가능하거나 부산물
[확인 후 삭제] : 실험 산출물/체크포인트 — 필요한 것만 남기고 지워야 함
[주의/보류]   : 현재 실행 중이거나, 뭔지 애매해서 직접 열어봐야 하는 것
```

---

## 1. 데이터 필터링 (`ocr_file_filter/`, `ocr_filter_result/`)

| 경로 | 크기 | 수정일 | 상태 |
|---|---|---|---|
| `ocr_file_filter/venv/` | 586M | 07-21 | [삭제 안전] pip venv, `pip install -r requirements.txt`로 재생성 가능 |
| `ocr_file_filter/labeler/.pytest_cache` | 36K | 07-14 | [삭제 안전] |
| `ocr_file_filter/**/__pycache__` (여러 곳) | 수백K | - | [삭제 안전] `find ocr_file_filter -name __pycache__ -exec rm -rf {} +` |
| `ocr_file_filter/_work/serve/*.log` | 100M+ (특히 `wk_paddle_g7_p8080.log` 47M, `external_b.log` 44M) | 07-17 | [확인 후 삭제] 서버 기동 로그, 디버깅 끝났으면 삭제 |
| `ocr_file_filter/_work/texlive_setup*.log` | 소량 | 07-19 | [삭제 안전] 설치 로그 |
| `ocr_file_filter/_work/compare_ac`, `_work/review`, `_work/weekend`, `_work/hardfalse_review` | - | 07-15~20 | [확인 후 삭제] 실험 스냅샷 폴더들, 이름이 날짜/목적별로 계속 새로 생김 — 결론 난 것만 남기고 정리 대상 |
| `ocr_file_filter/nightly.log` | - | 07-16 | [확인 후 삭제] |
| `ocr_filter_result/images/` | - | 07-20 | [확인 후 삭제] `finaloutput/final_dataset.jsonl`이 최종 산출물이면 원본 `images/`는 중간 산출물일 수 있음 — 확인 필요 |

---

## 2. 파인튜닝 (`finetuning/`, `jhyeo_trash/` 안의 finetune 관련)

### 2-1. 체크포인트 (제일 큰 덩어리, 확인 필수)

`finetuning/vlm/output/` 전체가 **563G**로 홈 디렉토리 사용량의 절반 이상입니다.

| 경로 | 크기 | 수정일 | 상태 |
|---|---|---|---|
| `finetuning/vlm/output/e19_qwen35_9b_37818/student_sft/checkpoint-*` (약 60개, 각 2.7G) | 177G | 07-23 | [확인 후 삭제] e19는 학습 끝난 것으로 보임 — 마지막/best 체크포인트 몇 개만 남기고 나머지 삭제 검토 |
| `finetuning/vlm/output/e20_qwen35_9b_pixel_bboxw/student_sft/checkpoint-*` (약 82개) | 285G | 07-26 | [확인 후 삭제] `final/` 폴더 있음 → 학습이 한 번 끝났던 런. resume 버전이 따로 있으니 이 런 자체가 중간에 끊긴 걸 수도 있음. 필요한 체크포인트만 남기기 |
| `finetuning/vlm/output/e20_qwen35_9b_pixel_bboxw_resume_20260726_233540/` | 30G | **07-27 (지금도 갱신 중)** | [주의/보류] **현재 진행 중인 학습으로 보임 (checkpoint-550이 몇 분 전 타임스탬프). 학습 끝나기 전엔 절대 건들지 마세요.** |
| `finetuning/vlm/output/e20_sweep_w2/`, `w3/`, `w5/`, 각 `_nobbox_baseline` | 각 12G (합 72G) | 07-2x | [확인 후 삭제] sweep 비교 실험, 결론 냈으면 로그/최종 지표만 남기고 체크포인트 삭제 검토 |
| `finetuning/vlm/output/e20_smoke/` | 3.5G | - | [확인 후 삭제] 이름상 스모크 테스트용 — 결과 확인 끝났으면 삭제 |

> tmux에서 확인용: `du -sh finetuning/vlm/output/*/student_sft/checkpoint-* | sort -V` 로 체크포인트 목록 보고, 필요한 것만 남기고 `rm -rf`.

### 2-2. 그 외

| 경로 | 크기 | 수정일 | 상태 |
|---|---|---|---|
| `finetuning/vlm/wandb/` | 132M | 07-26 | [확인 후 삭제] 오프라인 wandb 로그. `wandb_sync_loop.sh`로 이미 sync 됐으면 로컬 사본 삭제 가능 |
| `finetuning/vlm/.venv/` | 6.7G | 07-20 | [삭제 안전] requirements.txt로 재생성 가능 (단, 지금 학습이 이 venv를 쓰고 있으면 학습 끝난 뒤에) |
| `finetuning/vlm/_run_logs/` | - | 07-26 | [확인 후 삭제] 실행 로그 모음 |
| `finetuning/finetuning_dataset/__pycache__` | 8K | 07-24 | [삭제 안전] |
| `finetuning/finetuning_dataset/combined/*.bak`, `*.before_dotsocr_unify.bak` | - | - | [확인 후 삭제] 백업 파일, 최종본 확정됐으면 삭제 |

### 2-3. `jhyeo_trash/` (이름부터 trash, 통째로 검토 대상)

| 경로 | 크기 | 수정일 | 상태 |
|---|---|---|---|
| `jhyeo_trash/mineru_hybrid/mineru_env` | 8.4G | 07-10 | [삭제 안전] venv |
| `jhyeo_trash/mineru_work/venv` | 8.3G | 07-08 | [삭제 안전] venv |
| `jhyeo_trash/mineru_work/mineru_sft_env` | 8.3G | 07-09 | [삭제 안전] venv |
| `jhyeo_trash/mineru_work/eval` | 2.8G | 07-09 | [확인 후 삭제] |
| `jhyeo_trash/mineru_work/finetune` | 383M | 07-09 | [확인 후 삭제] |
| `jhyeo_trash/mineru_work/checkpoints` | 299M | 07-20 | [확인 후 삭제] |
| `jhyeo_trash/finetune_ko/datasets` | 1.3G | 07-20 | [확인 후 삭제] |
| `jhyeo_trash/finetune_ko/checkpoints` | 45M | 07-03 | [확인 후 삭제] |
| 나머지 `jhyeo_trash/**` | 합쳐서 수백M | 07-03~07-20 | [삭제 안전에 가까움] 마지막 수정이 7/20, 프로세스 사용 흔적 없음 (lsof 확인) — **폴더 전체를 한 번 열어보고 통째로 지우는 걸 권장** |

---

## 3. 평가 (`tsr_eval/`, `tsr_test/`, `finetuning/eval_318/`, `paddleOCRVL1.6/`)

여기가 **"실행 위치가 계속 바뀐다"**고 느끼시는 부분의 핵심입니다. 같은 걸 하는 폴더가 시간순으로 여러 개 생겼습니다.

### 3-1. `tsr_eval/` vs `tsr_test/` — 사실상 같은 일을 하는 앞/뒤 버전

- `tsr_eval/vlm/` (2.0M, 코드만) 은 `finetuning/vlm/` (67M, 코드+실행환경)의 **이전 버전**으로 보입니다. 파일 내용이 다르고(`collator.py` 등), `tsr_eval/vlm` 쪽 pycache 최종 수정이 07-07~09 이후로 멈춰있어 그 이후로는 안 건드린 걸로 보입니다.
  → `finetuning/vlm/`이 현재 쓰는 버전이면, `tsr_eval/vlm/` 전체(코드+venv_dpbench 등)는 [확인 후 삭제] 대상.
- `tsr_test/`는 07-23에 git clone된 걸로 보이는 최신 저장소(TFLOP 학습 코드, tsr_lable_tool, CLAUDE.md/AGENTS.md 포함)입니다. `tsr_eval/`가 이 저장소로 대체된 거라면 `tsr_eval/` 통째로 정리 대상.

| 경로 | 크기 | 상태 |
|---|---|---|
| `tsr_eval/venv_dpbench` | 294M | [삭제 안전] venv |
| `tsr_eval/logs_*.log` (서버/추론 로그 10여 개) | 수M | [확인 후 삭제] |
| `tsr_eval/gallery/`, `results/` | - | [확인 후 삭제] 옛 실험 갤러리/결과, `tsr_test`에 최신 버전 있으면 중복 |

### 3-2. `genos500_dpbench` 벤치마크가 3곳에 복사되어 있음

| 경로 | 크기 | 수정일 |
|---|---|---|
| `tsr_test/genos500_dpbench/` | 299M | 07-23 (최신, `cloudflared.log` 07-26까지 갱신) |
| `finetuning/eval_318/tsr_test_eval/genos500_dpbench/` | 353M | 07-23 |
| `tsr_eval/dp_bench/genos500_dpbench/` | 27M | 07-07 (내용 구조가 달라서 이건 다른 스냅샷일 수 있음) |

→ **어느 게 "지금 쓰는" 원본인지 정하고 나머지 2개는 지우거나 심볼릭 링크로 바꾸는 걸 권장.** 계속 새 프로젝트 폴더 만들 때마다 이 벤치마크를 통째로 복사해오는 패턴이 반복된 것 같습니다.

### 3-3. dpbench용 venv가 4벌

| 경로 | 크기 |
|---|---|
| `tsr_eval/venv_dpbench` | 294M |
| `finetuning/eval_318/venv_dpbench` | 162M |
| `finetuning/eval_318/_dpbench_venv` | 67M |
| `finetuning/eval_318/_ppb_venv` | - (pdf_parse_bench용, 별개일 수 있음) |

→ 지금 쓰는 것 1개만 남기고 나머지 삭제 검토. 전부 재생성 가능.

### 3-4. `paddleOCRVL1.6/`

전체가 07-03 생성, 마지막 스크립트 수정이 07-13. 지금 "평가" 워크플로우에서 안 쓰고 있다면:

| 경로 | 크기 | 상태 |
|---|---|---|
| `paddleOCRVL1.6/venv_paddle` | 8.3G | [삭제 안전] |
| `paddleOCRVL1.6/venv_vllm` | 9.0G | [삭제 안전] |
| `paddleOCRVL1.6/models/PaddleOCR-VL-1.6` | 1.8G 중 대부분 | [확인 후 삭제] 모델 가중치, 다시 안 쓰면 삭제 (재다운로드 필요) |

### 3-5. `finetuning/eval_318/`

| 경로 | 크기 | 상태 |
|---|---|---|
| `_infer_shards`, `_infer_shards_e20` | - | [확인 후 삭제] 샤드별 추론 중간 산출물, 최종 merge 결과 있으면 삭제 가능 |
| `omni_run_cdm_new63541.log` | - | [확인 후 삭제] |

---

## 4. 손대지 말아야 할 것 (0701 이전 / 현재 진행 중)

- `finetuning/vlm/.bak_20260617_135125`, `tsr_eval/vlm/.bak_20260617_135125` — 0701 이전 (06-17)
- `finetuning/vlm/{data,docs,prompt,requirements.txt,run_experiments.sh,script.sh}` 기본 골격 — 0701 이전 (06-29)
- `tsr_eval/vlm/{data,docs,prompt,requirements.txt,run_experiments.sh,script.sh,config}` 기본 골격 — 0701 이전 (06-29)
- **`finetuning/vlm/output/e20_qwen35_9b_pixel_bboxw_resume_20260726_233540/`** — 지금 이 순간도 체크포인트가 쌓이고 있는 학습 진행 중 폴더. 학습 끝나거나 중단한 뒤에만 정리.

---

## 다음 액션 제안 (직접 tmux에서 실행)

1. `jhyeo_trash/` 폴더 한 번 열어보고 통째로 `rm -rf` (제일 확실한 30G)
2. 안 쓰는 venv들부터 지우기 (venv_paddle, venv_vllm, tsr_eval/venv_dpbench, finetuning/vlm/.venv 등 — 다 합치면 수십G, 재생성만 하면 됨)
3. `tsr_eval/` 전체가 `tsr_test/` + `finetuning/vlm/`으로 대체된 게 맞는지 확인 → 맞으면 `tsr_eval/` 통째로 삭제 (venv_dpbench 포함 약 4G + 코드)
4. `genos500_dpbench` 3벌 중 최신 1개만 남기고 정리
5. `finetuning/vlm/output/` 체크포인트: e19, e20(첫 런), sweep 4종 — 각 폴더 들어가서 필요한 체크포인트 (예: 마지막 것 + best) 확인 후 나머지 `rm -rf` → 여기가 가장 큰 절감(수백G)이지만 제일 신중해야 함
