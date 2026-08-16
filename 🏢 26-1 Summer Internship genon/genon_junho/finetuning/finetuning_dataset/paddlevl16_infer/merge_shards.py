#!/usr/bin/env python3
"""`{split}.shard*.jsonl` 을 manifest 순서대로 `{split}.jsonl` 하나로 합친다.

manifest 의 모든 key 가 나왔는지 같이 확인한다. 빠진 게 있으면 run_all.sh 를 다시
돌리면 된다(이미 있는 key 는 건너뛴다).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
MANIFEST = Path("/home/jhyeo/finetuning/finetuning_dataset/gt_table_audit/manifest.jsonl")
SPLITS = ("train", "valid", "test")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE))
    ap.add_argument("--manifest", default=str(MANIFEST), help="manifest.jsonl 경로")
    ap.add_argument("--keep-shards", action="store_true", help="합친 뒤 샤드 파일을 지우지 않는다")
    args = ap.parse_args()
    d = Path(args.dir)

    order = {s: [] for s in SPLITS}
    with Path(args.manifest).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            order[r["split"]].append(r["key"])

    ok = True
    for split in SPLITS:
        shards = sorted(d.glob(f"{split}.shard*.jsonl"))
        merged = d / f"{split}.jsonl"
        rows = {}
        for p in list(shards) + ([merged] if merged.exists() else []):
            with p.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rows[r["key"]] = r

        keys = order[split]
        missing = [k for k in keys if k not in rows]
        with merged.open("w", encoding="utf-8") as f:
            for k in keys:
                if k in rows:
                    f.write(json.dumps(rows[k], ensure_ascii=False) + "\n")

        n_tbl = sum(r["pred_n_tables"] for r in rows.values())
        match = sum(1 for k in keys if k in rows and rows[k]["gt_n_tables"] == rows[k]["pred_n_tables"])
        print(f"[{split}] {len(rows)}/{len(keys)} 페이지, 예측 표 {n_tbl}개, "
              f"GT 와 표 개수 일치 {match} → {merged}")
        if missing:
            ok = False
            print(f"  빠진 {len(missing)}건: {missing[:5]}{' ...' if len(missing) > 5 else ''}")

        if not args.keep_shards:
            for p in shards:
                p.unlink()

    print("\n모두 채워짐" if ok else "\n빠진 페이지가 있다. run_all.sh 를 다시 돌릴 것.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
