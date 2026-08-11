# 필요한 파일 목록

순서: **필터링 및 자동 레이블링 파이프라인 → 파인튜닝 데이터 확정 → 파인튜닝 → 평가**

```
[1] ocr_file_filter          원본 → 필터링 + 자동 레이블링
        │
        ▼
[2] finetuning                데이터셋 확정 + 파인튜닝 실행
        │
        ▼
[3] eval_318 / tsr_test       dp500 / tsr200 평가
```

---

## 1. 필터링 & 자동 레이블링 — `/home/jhyeo/ocr_file_filter`

- **input**: `/home/jhyeo/ocr_filter_result`에 두고 진행.
- **실행**: `ocr_file_filter/scripts/run_pipeline.sh` — input/output 경로만 바꿔주면 재사용 가능.

## 2. 파인튜닝 — `/home/jhyeo/finetuning`

### 2-1. 데이터셋 — `finetuning/finetuning_dataset`

파인튜닝할 목록을 train/val/test로 나눠 저장. 버전별로 진행 중이며, 버전별 변화 로직은 txt로 기록.

**데이터셋 시각화**: `/home/jhyeo/result_rendering/app.py`를 포트에 서빙해서 띄우고, json 파일의 절대경로를 붙여넣으면 bbox + html 렌더링으로 시각화해서 볼 수 있음(파인튜닝 데이터셋뿐 아니라 eval용 추론 json도 지원).

> 오류가 나면 대부분 이미지 파일이 안 잡히는 경우 — 화면 제일 좌측 박스에 원본 이미지 경로를 절대경로로 붙여넣을 것.

### 2-2. 학습 실행 — `finetuning/vlm`

실제 파인튜닝 실행 코드(`config/*.yaml`, `run_experiments.sh` 등). `jhshin/tsr_test/train/vlm`을 읽기전용으로 복사해온 것, 상세는 `finetuning/README.md` 참고.

## 3. 평가 — `finetuning/eval_318`, `/home/jhyeo/tsr_test`

dp500 및 tsr200 평가 파이프라인.

- `tsr_test/instruction_set.md`의 명령어를 그대로 따라 실행해도 되고, 이 파일 자체가 프롬프트화되어 있어서 학습이 끝나면 이 md를 Claude 등에 태그하고 "학습한 버전의 어느 ckpt들 평가 돌려줘"라고 명령하면 같은 md에 테이블로 결과가 업데이트됨(기존 버전은 제일 잘 나온 것 하나만 남김).
- 평가 추론 output: `finetuning/vlm/eval_results`에 저장됨.

> **참고**: `/home/jhyeo/tsr_eval`은 `tsr_test`의 다른 목적 사본이라 이 파이프라인과 무관 — 헷갈리지 말 것.
