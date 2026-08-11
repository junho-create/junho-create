"""tsr200 rescored 예측을 complexity(simple/medium/complex/complex_col/complex_mix/
complex_row)별로 묶어 평균 TEDS(등)를 낸다. LLM-Judge 없이 TEDS만으로 빠르게
난이도 구간별 성능을 보고 싶을 때 사용한다.

전제: `eval/rescore_unified.py`로 만든 *_rescored/predictions_unified.jsonl
(각 레코드에 index/teds/teds_structure/span_f1/attribute_accuracy 포함)과
tsr200 매니페스트(`_infer_shards/tsr200_html.jsonl`, 각 레코드에 complexity 필드)를
index로 조인한다.

사용 예::

    python -m eval.complexity_breakdown \
        --tsr200 /home/jhyeo/finetuning/eval_318/_infer_shards/tsr200_html.jsonl \
        --metric teds \
        e24_ckpt1450=eval_results/e24_sweep_tsr200_ckpt1450_WITHOCR_rescored/predictions_unified.jsonl \
        e25_ckpt3900=eval_results/e25_sweep_tsr200_ckpt3900_WITHOCR_rescored/predictions_unified.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

_ORDER = ["simple", "medium", "complex", "complex_col", "complex_mix", "complex_row"]


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_complexity(tsr200_path: str) -> dict[int, str]:
    return {i: rec.get("complexity", "unknown") for i, rec in enumerate(_read_jsonl(tsr200_path))}


def breakdown(predictions_path: str, complexity: dict[int, str], metric: str) -> dict:
    vals: dict[str, list[float]] = defaultdict(list)
    for rec in _read_jsonl(predictions_path):
        idx = rec.get("index")
        if idx is None or idx not in complexity or metric not in rec:
            continue
        vals[complexity[idx]].append(rec[metric])

    row = {}
    all_vals: list[float] = []
    for bucket in _ORDER:
        v = vals.get(bucket, [])
        row[bucket] = sum(v) / len(v) if v else None
        all_vals += v
    row["overall"] = sum(all_vals) / len(all_vals) if all_vals else None
    row["n"] = len(all_vals)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="tsr200 complexity별 TEDS breakdown (LLM-Judge 없음)")
    ap.add_argument("--tsr200", required=True, help="tsr200_html.jsonl 경로 (complexity 필드 조인용)")
    ap.add_argument("--metric", default="teds", help="teds | teds_structure | span_f1 | attribute_accuracy")
    ap.add_argument("runs", nargs="+", help="tag=predictions_unified.jsonl(rescored) 쌍")
    args = ap.parse_args()

    complexity = _load_complexity(args.tsr200)

    header = f"{'run':22s} {'n':>4s} {'overall':>8s} " + " ".join(f"{b:>12s}" for b in _ORDER)
    print(header)
    for pair in args.runs:
        tag, path = pair.split("=", 1)
        row = breakdown(path, complexity, args.metric)
        cells = " ".join(
            f"{(row[b] if row[b] is not None else float('nan')):>12.4f}" for b in _ORDER
        )
        overall = row["overall"]
        print(f"{tag:22s} {row['n']:>4d} {overall:.4f}   {cells}" if overall is not None else f"{tag:22s} no data")


if __name__ == "__main__":
    main()
