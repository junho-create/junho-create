# finetuning_dataset

Qwen3.5-9B(Chandra div-HTML SFT) 파인튜닝용 gt_html 데이터셋 두 묶음. **이미지는 복사하지 않고
상대경로/심볼릭링크로 원본을 그대로 참조**한다 — `jhyeo/` 아래 상대 위치가 유지되는 한(예: 서버를
옮기더라도 `jhyeo/` 통째로 같이 이동) 이 폴더만 있어도 이미지까지 정상적으로 열린다.

## reference_16886/ — 기존 레퍼런스 학습셋 (16,886건, 사람 GT)

- `data/{train,valid,test}.jsonl` (+ `_layout`/`_table` 분할 버전) — 원본
  `chandra_table_layout_divhtml_16886/data/`를 그대로 복사(내용 무수정).
- `images` — 심볼릭링크 → `../../../jhyeo_trash/mineru_work/finetune/chandra_table_layout_divhtml_16886/images`
  (원본과 동일한 `{data,images}` 형제 구조를 그대로 재현했기 때문에, jsonl 안의
  `"image_path": "images/layout/xxx.png"`가 수정 없이 그대로 동작함).
- `chandra_convert_summary.json`/`inject_ocr_summary.json` — 원본 변환 시 남긴 통계, 참고용.

## new_63541/ — 이번에 새로 큐레이션한 데이터셋 (20,932건, resolved=True + 카테고리 정규화/깨진 표 제외 + Easy tier 제외 후 Medium+Hard만)

- `gt_html_dataset.jsonl` — `ocr_file_filter/filtering_result/gt_html_dataset.jsonl`(원본)을
  복사하면서 `image_path`만 **절대경로 → 이 폴더 기준 상대경로**로 재작성함
  (`../../ocr_filter_result/images/...`). 원본(`ocr_file_filter/filtering_result/`)은 그대로 두고
  안 건드림 — `python -m ocr_filter.cli export gt-html`로 언제든 재생성 가능한 파생 산출물.
- 생성 로직: 63,541건 중 `resolved==True`만 포함하고(Easy 35,621 + Medium 3,372 + Hard 19,054=58,047), 그중 복구불가능하게 깨진 표가 있는 1,494페이지(전부 Hard)를 더 제외해 56,553건,
  `ocr_file_filter/README.md`의 `[7] SFT 학습 포맷 변환` 참고.

## 합쳐서 SFT 할 때

두 jsonl(레퍼런스 9종 분할 중 필요한 것 + `new_63541/gt_html_dataset.jsonl`)을 원하는 비율로
concat/셔플해서 학습 config에 넣으면 된다 — 스키마 동일(`image_path, gt_html, prompt_style,
bbox_scale, output_format, ocr_info`). 분할(train/valid/test)은 여기서 안 함 — `new_63541`은
아직 안 나뉜 통짜 파일이라, 필요하면 `tsr_test/train/vlm/data/split_jsonl.py`(또는
`split_dataset.py`) 같은 기존 스크립트로 나눠 쓰면 된다.





명령어:

cd /home/jhyeo/finetuning/vlm
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0,2,3 \
NUM_GPUS=3 \
MASTER_PORT=29650 \
WANDB_MODE=online \
WANDB_PROJECT="tsr_vlm_train" \
WANDB_ENTITY="aisearch260330-genon" \
CONFIG=config/e19_qwen35_9b_37818.yaml \
PHASE=sft \
bash distill/run_distill.sh

