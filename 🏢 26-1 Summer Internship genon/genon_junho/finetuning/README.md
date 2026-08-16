# finetuning

Qwen3.5-9B(Chandra div-HTML SFT)를 실제로 파인튜닝하기 위한 데이터 + 코드 묶음.

```
finetuning/
├── finetuning_dataset/   데이터 (이미지는 복사 없이 상대경로/심볼릭링크로 원본 참조)
│   ├── reference_16886/  기존 레퍼런스(사람 GT), data/*.jsonl + images 심볼릭링크
│   ├── new_63541/        ocr_file_filter 큐레이션(20,932건, resolved=True + 정규화/필터링 후 Medium+Hard만) + train/valid/test 분할본
│   ├── combined/         위 둘을 레퍼런스 비율(≈83:10:7)로 합친 최종 학습셋
│   │                     (train 31,328 / valid 3,695 / test 2,795 = 37,818건)
│   │                     image_path 는 합칠 때 전부 절대경로로 고정(두 데이터셋이 이미지 루트가
│   │                     서로 달라 상대경로 한 벌로는 둘 다 못 맞춰서)
│   └── README.md         데이터셋 상세
└── vlm/                  학습 코드(jhshin/tsr_test/train/vlm 을 읽기전용으로 복사, 원본 미변경)
    ├── config/exp_chandra_combined_37818.yaml   ← combined/ 데이터셋용으로 새로 만든 config
    ├── config/exp_*.yaml                        원본 config들(jhshin 절대경로 그대로, 참고용)
    ├── distill/, train/, utils/, data/, eval/, scripts/, prompt/, docs/
    └── (output/, _train_data/, eval_data/, eval_results/, wandb/, logs/ 는 무거워서 안 가져옴)
```

## 학습 실행

```bash
cd /NHNHOME/WORKSPACE/0426030039_A/jhyeo/finetuning/vlm
source <venv>/bin/activate   # transformers/peft/deepspeed 등, requirements.txt 참고

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_GPUS=4 \
MASTER_PORT=29650 \
WANDB_MODE=online \
CONFIG=config/exp_chandra_combined_37818.yaml \
PHASE=sft \
bash distill/run_distill.sh
```

`exp_chandra_combined_37818.yaml`은 원본 `exp_chandra_all_e18align.yaml`(jhshin 소유, 절대경로
하드코딩)을 베이스로 경로만 이 프로젝트 기준으로 바꾼 파생 config다 — 하이퍼파라미터(LoRA
r=128/alpha=256, lr=1.5e-5, epoch=7 등)는 원본과 동일. 베이스 모델은 `jhyeo/ocr_file_filter/_models/
Qwen3.5-9B`(이미 로컬에 있음)를 가리키므로 shkim/jhshin 디렉토리 없이도 동작한다.

## 주의

- `finetuning_dataset/combined/*.jsonl`은 `.gitignore` 처리 대상(용량 큼) — 서버 이전 시 파일
  그대로 복사하거나, 위 병합 스크립트를 다시 돌려 재생성한다(스크립트 자체는 이 세션 대화 로그에만
  있고 별도 파일로 안 남겨뒀음 — 필요하면 재작성 가능: reference_16886/data/*.jsonl +
  new_63541/*.jsonl 의 image_path를 각자의 데이터셋 루트 기준 절대경로로 바꿔서 concat).
- `vlm/`의 원본 config들(`exp_chandra_all_e18align.yaml` 등)은 jhshin 소유 절대경로를 그대로
  가지고 있어 이 프로젝트만으로는 안 돌아간다 — 참고용으로만 남겨둠, 실제로 쓸 건
  `exp_chandra_combined_37818.yaml`.
- 학습 코드 자체(`train_qlora.py`, `distill/*.py` 등)는 원본 그대로라 무거운 의존성(torch/
  transformers/deepspeed/vllm)은 GPU/CUDA 환경에 맞춰 별도 설치 필요(`vlm/requirements.txt` 참고).
