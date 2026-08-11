"""
실험 결과 집계 및 비교 표 출력

지정 디렉토리 하위의 모든 metrics.json을 자동 탐색하여 비교 테이블을 생성한다.

Usage:
    python -m eval.aggregate_results --results_dir eval_results/
    python -m eval.aggregate_results --results_dir eval_results/ --output comparison.json
    python -m eval.aggregate_results --results_dir eval_results/ --sort_by avg_teds
"""

import argparse
import json
import os
from pathlib import Path

METRIC_KEYS = [
    ("avg_teds", "TEDS"),
    ("avg_teds_structure", "TEDS-S"),
    ("avg_teds_norm", "TEDS-N"),
    ("avg_teds_norm_structure", "TEDS-NS"),
    ("avg_span_f1", "Span F1"),
    ("avg_attribute_accuracy", "Attr Acc"),
    ("avg_gca", "GCA"),
    ("avg_gsa", "GSA"),
    ("simple_teds", "Simple"),
    ("medium_teds", "Medium"),
    ("complex_teds", "Complex"),
]


def scan_metrics(results_dir: str) -> dict[str, dict]:
    """results_dir 하위의 모든 metrics.json을 자동 탐색한다."""
    loaded = {}
    root = Path(results_dir)
    if not root.exists():
        return loaded

    for metrics_path in sorted(root.rglob("metrics.json")):
        # 디렉토리명을 실험명으로 사용
        rel = metrics_path.parent.relative_to(root)
        experiment_name = str(rel) if str(rel) != "." else metrics_path.parent.name
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                loaded[experiment_name] = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: {metrics_path} 로드 실패 ({e})")
    return loaded


def format_value(v) -> str:
    """메트릭 값을 소수점 4자리로 포맷."""
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def print_comparison_table(loaded: dict[str, dict], sort_by: str = ""):
    """비교 표를 콘솔에 출력한다."""
    if not loaded:
        print("No experiment results found.")
        return

    # 정렬
    names = list(loaded.keys())
    if sort_by and sort_by in {k for k, _ in METRIC_KEYS}:
        names.sort(
            key=lambda n: loaded[n].get(sort_by, -1)
            if loaded[n] is not None else -1,
            reverse=True,
        )

    # 실험명 최대 폭 계산
    name_width = max(len(n) for n in names)
    name_width = max(name_width, 10)  # 최소 10

    headers = ["Experiment"]
    widths = [name_width]
    for _, label in METRIC_KEYS:
        headers.append(label)
        widths.append(max(len(label), 9))

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def fmt_row(vals):
        parts = []
        for v, w in zip(vals, widths):
            parts.append(f" {str(v):<{w}} ")
        return "|" + "|".join(parts) + "|"

    print(sep)
    print(fmt_row(headers))
    print(sep)

    for name in names:
        metrics = loaded[name]
        row = [name]
        for key, _ in METRIC_KEYS:
            val = metrics.get(key) if metrics else None
            row.append(format_value(val))
        print(fmt_row(row))

    print(sep)


def build_comparison_json(loaded: dict[str, dict]) -> dict:
    """비교 결과를 JSON 구조로 생성한다."""
    rows = []
    for name, metrics in loaded.items():
        row = {"experiment_id": name}
        for key, label in METRIC_KEYS:
            row[key] = metrics.get(key) if metrics else None
        # 부가 정보
        if metrics:
            row["total_samples"] = metrics.get("total_samples")
            row["backend"] = metrics.get("backend")
        rows.append(row)
    return {"experiments": rows, "metric_keys": [k for k, _ in METRIC_KEYS]}


def main():
    parser = argparse.ArgumentParser(description="실험 결과 집계")
    parser.add_argument(
        "--results_dir",
        default="eval_results/",
        help="eval_results 루트 디렉토리",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="비교 표 JSON 출력 경로 (미지정 시 콘솔만 출력)",
    )
    parser.add_argument(
        "--sort_by",
        default="",
        help="정렬 기준 메트릭 (예: avg_teds, avg_span_f1)",
    )
    args = parser.parse_args()

    loaded = scan_metrics(args.results_dir)

    print(f"Found {len(loaded)} experiment(s) in {args.results_dir}")
    print()

    print_comparison_table(loaded, sort_by=args.sort_by)

    if args.output:
        comparison = build_comparison_json(loaded)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        print(f"\nSaved comparison table: {output_path}")


if __name__ == "__main__":
    main()
