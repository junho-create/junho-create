"""
Merge multiple shard evaluation runs into one combined result.

Usage:
  python -m eval.merge_shard_runs \
    --run_glob "eval_results/serving_api_model_*_shard*_..." \
    --output_dir eval_results/serving_api_model_merged_...
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any

from eval.metrics import compute_aggregate_metrics


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _shard_sort_key(run_dir: Path) -> tuple[int, str]:
    m = re.search(r"shard(\d+)", run_dir.name)
    if not m:
        return (10_000_000, run_dir.name)
    return (int(m.group(1)), run_dir.name)


def _complexity_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for comp in ("simple", "medium", "complex"):
        items = [p for p in predictions if str(p.get("complexity", "unknown")) == comp]
        n = len(items)
        if n == 0:
            rows.append(
                {
                    "complexity": comp,
                    "n": 0,
                    "avg_teds": 0.0,
                    "avg_teds_structure": 0.0,
                    "avg_span_f1": 0.0,
                    "avg_attribute_accuracy": 0.0,
                    "avg_inference_time": 0.0,
                }
            )
            continue
        rows.append(
            {
                "complexity": comp,
                "n": n,
                "avg_teds": sum(float(x.get("teds", 0.0)) for x in items) / n,
                "avg_teds_structure": sum(float(x.get("teds_structure", 0.0)) for x in items) / n,
                "avg_span_f1": sum(float(x.get("span_f1", 0.0)) for x in items) / n,
                "avg_attribute_accuracy": sum(float(x.get("attribute_accuracy", 0.0)) for x in items) / n,
                "avg_inference_time": sum(float(x.get("inference_time", 0.0)) for x in items) / n,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge shard eval runs")
    parser.add_argument("--run_glob", required=True, help="Glob for shard run directories")
    parser.add_argument("--output_dir", required=True, help="Merged output directory")
    parser.add_argument("--expected_total", type=int, default=0, help="Optional expected merged sample count")
    args = parser.parse_args()

    run_dirs = [Path(p) for p in glob.glob(args.run_glob)]
    run_dirs = sorted(run_dirs, key=_shard_sort_key)
    if not run_dirs:
        raise SystemExit(f"No runs matched: {args.run_glob}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged: list[dict[str, Any]] = []
    merged_metrics_meta: dict[str, Any] = {}
    used_runs: list[str] = []

    for run_dir in run_dirs:
        pred_path = run_dir / "predictions.jsonl"
        met_path = run_dir / "metrics.json"
        if not pred_path.exists() or not met_path.exists():
            # Only include fully completed shard runs.
            continue
        preds = _read_jsonl(pred_path)
        if not preds:
            continue
        merged.extend(preds)
        used_runs.append(str(run_dir))
        if not merged_metrics_meta:
            merged_metrics_meta = _load_json(met_path)

    if not merged:
        raise SystemExit("No completed shard runs with predictions/metrics were found.")

    if args.expected_total > 0 and len(merged) != args.expected_total:
        print(
            f"Warning: merged sample count mismatch: expected={args.expected_total}, got={len(merged)}"
        )

    agg = compute_aggregate_metrics(merged)
    metrics = agg.to_dict()
    metrics["avg_inference_time"] = sum(float(x.get("inference_time", 0.0)) for x in merged) / len(merged)
    for key in ("backend", "batch_size", "max_new_tokens", "gt_source", "strict_aihub_gt"):
        if key in merged_metrics_meta:
            metrics[key] = merged_metrics_meta[key]

    pred_out = output_dir / "predictions.jsonl"
    with pred_out.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics_out = output_dir / "metrics.json"
    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    comp_rows = _complexity_rows(merged)
    comp_json = output_dir / "complexity_metrics.json"
    with comp_json.open("w", encoding="utf-8") as f:
        json.dump(comp_rows, f, ensure_ascii=False, indent=2)

    comp_tsv = output_dir / "complexity_metrics.tsv"
    with comp_tsv.open("w", encoding="utf-8") as f:
        f.write(
            "complexity\tn\tavg_teds\tavg_teds_structure\tavg_span_f1\tavg_attribute_accuracy\tavg_inference_time\n"
        )
        for row in comp_rows:
            f.write(
                f"{row['complexity']}\t{row['n']}\t{row['avg_teds']:.6f}\t"
                f"{row['avg_teds_structure']:.6f}\t{row['avg_span_f1']:.6f}\t"
                f"{row['avg_attribute_accuracy']:.6f}\t{row['avg_inference_time']:.6f}\n"
            )

    used_runs_out = output_dir / "merged_from_runs.txt"
    with used_runs_out.open("w", encoding="utf-8") as f:
        for run in used_runs:
            f.write(run + "\n")

    print(f"Merged samples: {len(merged)}")
    print(f"Saved: {pred_out}")
    print(f"Saved: {metrics_out}")
    print(f"Saved: {comp_json}")
    print(f"Saved: {comp_tsv}")
    print(f"Saved: {used_runs_out}")


if __name__ == "__main__":
    main()
