#!/usr/bin/env python3
"""PaddleOCR-VL 1.6 으로 combined_e24_refined 의 표 포함 layout 페이지를 추론한다.

대상은 gt_table_audit/manifest.jsonl 의 5,030 페이지(train 4,101 / valid 473 / test 456)
— TSR crop(`unified_table_with_ocr`)은 제외하고 `data-label="Table"` div 가 1개 이상인
`unified_layout` 페이지만이다.

VL 인식은 native backend 가 페이지당 20분 이상 걸려 쓸 수 없어서, vLLM OpenAI 서버에
붙인다(`serve_vllm.sh` 로 먼저 띄울 것). 레이아웃 검출(PP-DocLayoutV3)만 로컬 GPU 에서
돈다.

클라이언트(이미지 crop·인코딩·블록 조립)가 CPU 단일 스레드라 워커 하나로는 vLLM 을
25% 밖에 못 채운다. `--shard i/N` 으로 여러 프로세스를 띄워 같은 서버에 물리면 거의
선형으로 빨라진다. 샤드마다 `{split}.shard{i}.jsonl` 에 따로 쓰고(같은 파일에 여러
프로세스가 append 하면 긴 줄이 섞인다) 끝나고 `merge_shards.py` 로 합친다.

split 별로 append 하고, 이미 기록된 key 는 건너뛴다(재개 가능).

사용:
    ./serve_vllm.sh 0 8118                    # 먼저 서버
    ./run_all.sh                              # 샤드 6개로 전체
    python3 run_infer.py --split test --limit 32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
MANIFEST = Path("/home/jhyeo/finetuning/finetuning_dataset/gt_table_audit/manifest.jsonl")
SPLITS = ("train", "valid", "test")


def load_manifest(manifest_path, splits, limit=None):
    by_split = {s: [] for s in splits}
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] in by_split:
                by_split[r["split"]].append(r)
    if limit:
        by_split = {s: v[:limit] for s, v in by_split.items()}
    return by_split


def done_keys(path: Path) -> set[str]:
    """이미 기록된 key. 마지막 줄이 잘려 있으면 무시한다(중단된 run 대비)."""
    keys = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


def norm_bbox(bbox, w, h):
    """픽셀 bbox → GT 와 같은 0-1000 정규화 좌표. GT 는 bbox_scale=1000 이다."""
    if not bbox or w <= 0 or h <= 0:
        return None
    x0, y0, x1, y1 = bbox
    return [
        round(x0 * 1000 / w),
        round(y0 * 1000 / h),
        round(x1 * 1000 / w),
        round(y1 * 1000 / h),
    ]


def predict_one(pipe, img, tries=4):
    """페이지 1장을 재시도하며 추론한다.

    vLLM 이 동시 요청을 받으면 HF fast tokenizer 가 `Already borrowed` 400 을 간헐적으로
    뱉는다. 입력 문제가 아니라 경합이라 잠깐 쉬었다 다시 보내면 대개 통과한다.
    """
    for attempt in range(tries):
        try:
            return next(iter(pipe.predict(img)))
        except Exception as e:
            if attempt == tries - 1:
                print(f"    skip {img}: {type(e).__name__}: {e}", flush=True)
                return None
            time.sleep(2 * (attempt + 1))
    return None


def pack(rec: dict, res_json: dict) -> dict:
    """페이지 1건의 예측을 저장 형태로 만든다."""
    w, h = res_json.get("width"), res_json.get("height")
    blocks = []
    for b in res_json.get("parsing_res_list", []):
        bbox = b.get("block_bbox")
        blocks.append({
            "label": b.get("block_label"),
            "bbox": bbox,
            "bbox_1000": norm_bbox(bbox, w, h),
            "order": b.get("block_order"),
            "content": b.get("block_content"),
        })
    tables = [
        {"index": i, "bbox": b["bbox"], "bbox_1000": b["bbox_1000"], "html": b["content"]}
        for i, b in enumerate(
            (b for b in blocks if b["label"] == "table"), start=1
        )
    ]
    det = res_json.get("layout_det_res") or {}
    return {
        "key": rec["key"],
        "split": rec["split"],
        "line_no": rec["line_no"],
        "image_path": rec["image_path"],
        "width": w,
        "height": h,
        "gt_n_tables": rec["n_tables"],
        "pred_n_tables": len(tables),
        "pred_tables": tables,
        "pred_blocks": blocks,
        "layout_boxes": [
            {"label": x.get("label"), "score": x.get("score"), "bbox": x.get("coordinate")}
            for x in det.get("boxes", [])
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(HERE))
    ap.add_argument("--split", choices=SPLITS, action="append",
                    help="반복 지정 가능. 생략하면 train/valid/test 전부")
    ap.add_argument("--limit", type=int, help="split 당 최대 건수 (디버그용)")
    ap.add_argument("--chunk", type=int, default=64, help="한 번에 predict 에 넘길 페이지 수")
    ap.add_argument("--server-url", default="http://127.0.0.1:8118/v1")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--device", default="gpu:0", help="레이아웃 검출용 로컬 GPU")
    ap.add_argument("--shard", default="0/1",
                    help="'i/N' — 이 프로세스가 맡을 샤드. 출력은 {split}.shard{i}.jsonl")
    ap.add_argument("--manifest", default=str(MANIFEST), help="manifest.jsonl 경로")
    args = ap.parse_args()

    shard_i, shard_n = (int(v) for v in args.shard.split("/"))
    if not 0 <= shard_i < shard_n:
        ap.error(f"잘못된 --shard {args.shard}")

    splits = args.split or list(SPLITS)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_split = load_manifest(Path(args.manifest), splits, args.limit)
    if shard_n > 1:
        # 라운드로빈이라 페이지 무게가 샤드마다 고르게 섞인다.
        by_split = {s: v[shard_i::shard_n] for s, v in by_split.items()}

    from paddleocr import PaddleOCRVL

    pipe = PaddleOCRVL(
        pipeline_version="v1.6",
        device=args.device,
        vl_rec_backend="vllm-server",
        vl_rec_server_url=args.server_url,
        vl_rec_max_concurrency=args.concurrency,
    )

    grand_t0 = time.time()
    for split in splits:
        recs = by_split[split]
        suffix = "" if shard_n == 1 else f".shard{shard_i}"
        out_path = out_dir / f"{split}{suffix}.jsonl"
        # 앞선 run 이 남긴 건 어느 샤드 파일에 있든 다시 하지 않는다.
        skip = set()
        for p in out_dir.glob(f"{split}*.jsonl"):
            skip |= done_keys(p)
        todo = [r for r in recs if r["key"] not in skip]
        print(f"\n[{split}] 전체 {len(recs)} / 완료 {len(skip)} / 남은 {len(todo)}", flush=True)
        if not todo:
            continue

        t0 = time.time()
        n_done = 0
        n_fail = 0
        with out_path.open("a", encoding="utf-8") as fout:
            for i in range(0, len(todo), args.chunk):
                batch = todo[i : i + args.chunk]
                imgs = [r["image_path"] for r in batch]
                try:
                    results = list(pipe.predict(imgs))
                except Exception as e:  # 청크 통째로 죽으면 페이지별로 다시
                    print(f"  chunk 실패({type(e).__name__}: {e}) → 페이지 단위 재시도", flush=True)
                    results = [predict_one(pipe, im) for im in imgs]

                # predict 는 입력 순서를 지키지만, 실패 자리에 None 이 섞일 수 있어
                # 길이가 맞을 때만 zip 한다.
                if len(results) != len(batch):
                    print(f"  결과 수 불일치 {len(results)} != {len(batch)}, 청크 건너뜀", flush=True)
                    n_fail += len(batch)
                    continue

                for rec, res in zip(batch, results):
                    if res is None:
                        n_fail += 1
                        continue
                    fout.write(json.dumps(pack(rec, res.json["res"]), ensure_ascii=False) + "\n")
                    n_done += 1
                fout.flush()
                os.fsync(fout.fileno())

                el = time.time() - t0
                rate = n_done / el if el else 0
                left = (len(todo) - n_done - n_fail) / rate if rate else 0
                print(
                    f"  [{split}] {n_done + n_fail}/{len(todo)}  "
                    f"{rate:.2f} page/s  경과 {el/60:.1f}분  ETA {left/60:.1f}분  실패 {n_fail}",
                    flush=True,
                )

        print(f"[{split}] 완료: 성공 {n_done} / 실패 {n_fail} → {out_path}", flush=True)

    print(f"\nALL DONE  총 {(time.time()-grand_t0)/60:.1f}분", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
