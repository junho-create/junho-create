#!/usr/bin/env python3
"""combined_e24_refined 의 layout 레코드에서 Table div 를 뽑아 manifest.jsonl 을 만든다.

TSR crop(`prompt_style == "unified_table_with_ocr"`)은 제외하고, layout 페이지
(`unified_layout`) 중 `data-label="Table"` div 가 1개 이상인 것만 대상으로 한다.
Table 라벨만 있는 layout 페이지(38장)도 포함한다.

정적 검사에서 이미 걸리는 것들(빈 표, `<table>` 없음, 극소 bbox)은 judge 를 태우지
않고 바로 사람 검수 큐로 보낼 수 있게 `static_flags` 에 기록해 둔다.

사용:
    python3 extract.py                       # 전체
    python3 extract.py --split train --limit 100
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup

DATASET_DIR = Path("/home/jhyeo/finetuning/finetuning_dataset/combined_e24_refined")
SPLITS = ("train", "valid", "test")

# bbox 면적이 페이지의 이 비율 미만이면 표로 보기 어렵다(p5 = 1.8%, 중앙값 = 14.9%).
TINY_BBOX_AREA = 0.01


def parse_tables(gt_html: str) -> list[dict]:
    """gt_html 에서 Table div 를 등장 순서대로 뽑는다.

    정규식 대신 파서를 쓴다 — 표 안에 중첩 <div> 가 있으면 non-greedy 정규식이
    엉뚱한 </div> 에서 끊긴다.
    """
    soup = BeautifulSoup(gt_html, "html.parser")
    tables = []
    for i, div in enumerate(soup.find_all("div", attrs={"data-label": "Table"}), start=1):
        raw_bbox = (div.get("data-bbox") or "").split()
        try:
            bbox = [int(float(v)) for v in raw_bbox]
        except ValueError:
            bbox = []
        if len(bbox) != 4:
            bbox = []
        inner = div.decode_contents().strip()
        tables.append({"index": i, "bbox": bbox, "html": inner})
    return tables


def static_flags(tables: list[dict]) -> list[str]:
    """judge 없이도 확실히 이상한 건들. `{사유}@{표번호}` 형식."""
    flags = []
    for t in tables:
        i = t["index"]
        html = t["html"]
        if not html:
            flags.append(f"empty_table@{i}")
            continue
        soup = BeautifulSoup(html, "html.parser")
        if soup.find("table") is None:
            flags.append(f"no_table_tag@{i}")
        elif not soup.find_all("tr"):
            flags.append(f"unparsable_html@{i}")
        bbox = t["bbox"]
        if not bbox:
            flags.append(f"bad_bbox@{i}")
        else:
            x0, y0, x1, y1 = bbox
            area = max(0, x1 - x0) * max(0, y1 - y0) / 1_000_000
            if area < TINY_BBOX_AREA:
                flags.append(f"tiny_bbox@{i}")
    return flags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=str(DATASET_DIR))
    ap.add_argument("--out", default=str(Path(__file__).parent / "manifest.jsonl"))
    ap.add_argument("--split", choices=SPLITS, help="한 split 만 처리")
    ap.add_argument("--limit", type=int, help="split 당 최대 건수 (디버그용)")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    splits = [args.split] if args.split else list(SPLITS)

    rows: list[dict] = []
    flag_counter: Counter[str] = Counter()
    ntables_counter: Counter[int] = Counter()

    for split in splits:
        path = dataset_dir / f"{split}.jsonl"
        kept = 0
        for lineno, line in enumerate(path.open(encoding="utf-8")):
            rec = json.loads(line)
            if rec.get("prompt_style") != "unified_layout":
                continue
            tables = parse_tables(rec["gt_html"])
            if not tables:
                continue
            if args.limit is not None and kept >= args.limit:
                break
            flags = static_flags(tables)
            flag_counter.update(f.split("@")[0] for f in flags)
            ntables_counter[len(tables)] += 1
            rows.append({
                "key": f"{split}_{lineno:06d}",
                "split": split,
                "line_no": lineno,
                "image_path": rec["image_path"],
                "n_tables": len(tables),
                "tables": tables,
                "static_flags": flags,
            })
            kept += 1
        print(f"[{split}] Table 포함 layout 페이지 {kept}건")

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_tbl = sum(r["n_tables"] for r in rows)
    n_flagged = sum(1 for r in rows if r["static_flags"])
    print(f"\n총 {len(rows)}페이지 / 표 {n_tbl}개 → {out}")
    print(f"정적 플래그가 붙은 페이지: {n_flagged}건")
    for name, cnt in flag_counter.most_common():
        print(f"  {name}: {cnt}개 표")
    print("페이지당 표 개수:", dict(sorted(ntables_counter.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
