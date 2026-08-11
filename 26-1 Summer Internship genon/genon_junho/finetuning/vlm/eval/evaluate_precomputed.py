"""
사전 생성 결과(precomputed JSONL) 평가 스크립트.

테스트 JSONL을 기준으로 조인 키(기본: image_path)로 pred JSONL을 매칭한 뒤,
GT/Pred HTML 필드(기본: gt_html)를 비교하여 평가 메트릭을 계산한다.

Usage:
    cd train/vlm
    python -m eval.evaluate_precomputed \
        --test_data _train_data/20260225_1_3000/data_split/test.jsonl \
        --pred_data _train_data/20260225_1_3000/data/train_ocr_claude.jsonl \
        --output_dir eval_results/precomputed_claude_test100_$(date +%Y%m%d_%H%M%S) \
        --join_key image_path \
        --gt_field gt_html \
        --pred_field gt_html
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import (
    TEDSCalculator,
    compute_aggregate_metrics,
    compute_grid_metrics,
    compute_span_metrics,
    normalize_complexity_label,
    to_base_complexity_bucket,
)
from utils.html_utils import annotate_empty_cells_with_token


def _normalize_empty_cells_html(
    html: str,
    enabled: bool,
    empty_cell_token: str,
) -> str:
    text = str(html or "").strip()
    if not text:
        return text
    if not enabled:
        return text
    token = str(empty_cell_token or "").strip()
    if not token:
        return text
    return annotate_empty_cells_with_token(text, empty_token=token)


def _load_jsonl(path: str, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_complexity(value: Any) -> str:
    comp = normalize_complexity_label(value)
    if comp in {"simple", "medium"}:
        return comp
    if comp.startswith("complex"):
        return comp
    return "unknown"


def _build_index(rows: list[dict[str, Any]], key: str) -> tuple[dict[str, dict[str, Any]], int]:
    index: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for row in rows:
        key_val = str(row.get(key, "")).strip()
        if not key_val:
            continue
        if key_val in index:
            duplicate_count += 1
        index[key_val] = row
    return index, duplicate_count


def _subset_metrics(
    rows: list[dict[str, Any]],
    join_key: str,
    gt_field: str,
    pred_field: str,
    test_data: str,
    pred_data: str,
) -> dict[str, Any]:
    agg = compute_aggregate_metrics(rows)
    metrics = agg.to_dict()
    metrics["join_key"] = join_key
    metrics["gt_field"] = gt_field
    metrics["pred_field"] = pred_field
    metrics["test_data"] = test_data
    metrics["pred_data"] = pred_data
    metrics["avg_inference_time"] = 0.0
    return metrics


def _write_complexity_artifacts(
    output_dir: str,
    predictions: list[dict[str, Any]],
    join_key: str,
    gt_field: str,
    pred_field: str,
    test_data: str,
    pred_data: str,
) -> None:
    base_order = ["simple", "medium", "complex"]
    detail_order = ["complex_nested", "complex_col", "complex_row", "complex_mix"]

    by_base: dict[str, list[dict[str, Any]]] = {k: [] for k in base_order}
    by_raw: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        raw_comp = _normalize_complexity(row.get("complexity", "unknown"))
        by_raw.setdefault(raw_comp, []).append(row)

        base_comp = to_base_complexity_bucket(raw_comp)
        if base_comp in by_base:
            by_base[base_comp].append(row)

    ordered_raw = [c for c in detail_order if c in by_raw]
    extra_raw = sorted(
        c for c in by_raw.keys()
        if c not in set(base_order) and c not in set(detail_order)
    )
    ordered_raw.extend(extra_raw)

    artifact_sets: list[tuple[str, list[dict[str, Any]]]] = []
    for comp in base_order:
        artifact_sets.append((comp, by_base[comp]))
    for comp in ordered_raw:
        artifact_sets.append((comp, by_raw[comp]))

    for comp, subset in artifact_sets:
        pred_path = os.path.join(output_dir, f"predictions_{comp}.jsonl")
        _write_jsonl(pred_path, subset)

        metrics = _subset_metrics(
            rows=subset,
            join_key=join_key,
            gt_field=gt_field,
            pred_field=pred_field,
            test_data=test_data,
            pred_data=pred_data,
        )
        metrics_path = os.path.join(output_dir, f"metrics_{comp}.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)


def evaluate_precomputed(
    test_data: str,
    pred_data: str,
    output_dir: str,
    join_key: str = "image_path",
    gt_field: str = "gt_html",
    pred_field: str = "gt_html",
    complexity_field: str = "complexity",
    max_samples: int | None = None,
    strict_missing: bool = False,
    normalize_empty_cells: bool = False,
    empty_cell_token: str = "__EMPTY__",
) -> dict[str, Any]:
    test_rows = _load_jsonl(test_data, max_samples=max_samples)
    pred_rows = _load_jsonl(pred_data)
    pred_index, pred_duplicate_count = _build_index(pred_rows, join_key)

    teds_calc = TEDSCalculator(structure_only=False)
    teds_struct_calc = TEDSCalculator(structure_only=True)
    teds_norm_calc = TEDSCalculator(structure_only=False, normalize=True)
    teds_norm_struct_calc = TEDSCalculator(structure_only=True, normalize=True)

    predictions: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    for i, test_row in enumerate(test_rows):
        key_val = str(test_row.get(join_key, "")).strip()
        if not key_val:
            missing_rows.append(
                {
                    "index": i,
                    "reason": "missing_join_key",
                    "join_key": join_key,
                    "test_row": test_row,
                }
            )
            continue

        pred_row = pred_index.get(key_val)
        if pred_row is None:
            missing_rows.append(
                {
                    "index": i,
                    "reason": "no_matching_pred_row",
                    "join_key": join_key,
                    "join_value": key_val,
                    "test_row": test_row,
                }
            )
            continue

        gt_html = str(test_row.get(gt_field, "") or "")
        pred_html = str(pred_row.get(pred_field, "") or "")
        gt_html_eval = _normalize_empty_cells_html(
            gt_html,
            enabled=normalize_empty_cells,
            empty_cell_token=empty_cell_token,
        )
        pred_html_eval = _normalize_empty_cells_html(
            pred_html,
            enabled=normalize_empty_cells,
            empty_cell_token=empty_cell_token,
        )
        complexity = _normalize_complexity(test_row.get(complexity_field, "unknown"))

        span = compute_span_metrics(pred_html_eval, gt_html_eval)
        grid = compute_grid_metrics(pred_html_eval, gt_html_eval)

        predictions.append(
            {
                "index": i,
                "image_path": test_row.get("image_path", ""),
                "image_path_raw": test_row.get("image_path", ""),
                "join_key": join_key,
                "join_value": key_val,
                "pred_html": pred_html,
                "pred_html_eval": pred_html_eval,
                "gt_html": gt_html,
                "gt_html_eval": gt_html_eval,
                "full_response": "",
                "complexity": complexity,
                "teds": teds_calc.compute(pred_html_eval, gt_html_eval),
                "teds_structure": teds_struct_calc.compute(pred_html_eval, gt_html_eval),
                "teds_norm": teds_norm_calc.compute(pred_html_eval, gt_html_eval),
                "teds_norm_structure": teds_norm_struct_calc.compute(pred_html_eval, gt_html_eval),
                "span_f1": span.span_f1,
                "span_precision": span.span_precision,
                "span_recall": span.span_recall,
                "position_f1": span.position_f1,
                "attribute_accuracy": span.attribute_accuracy,
                "colspan_accuracy": span.colspan_accuracy,
                "rowspan_accuracy": span.rowspan_accuracy,
                "gca": grid.gca,
                "gsa": grid.gsa,
                "inference_time": 0.0,
            }
        )

    if strict_missing and missing_rows:
        raise ValueError(
            f"strict_missing=True and {len(missing_rows)} rows were not matched. "
            f"Check output unmatched_test_rows.jsonl"
        )

    os.makedirs(output_dir, exist_ok=True)
    pred_out_path = os.path.join(output_dir, "predictions.jsonl")
    _write_jsonl(pred_out_path, predictions)

    unmatched_path = os.path.join(output_dir, "unmatched_test_rows.jsonl")
    _write_jsonl(unmatched_path, missing_rows)

    agg = compute_aggregate_metrics(predictions)
    metrics = agg.to_dict()
    metrics.update(
        {
            "join_key": join_key,
            "gt_field": gt_field,
            "pred_field": pred_field,
            "complexity_field": complexity_field,
            "test_data": test_data,
            "pred_data": pred_data,
            "test_total": len(test_rows),
            "pred_total": len(pred_rows),
            "matched_total": len(predictions),
            "unmatched_total": len(missing_rows),
            "pred_duplicate_key_count": pred_duplicate_count,
            "avg_inference_time": 0.0,
            "normalize_empty_cells": bool(normalize_empty_cells),
            "empty_cell_token": (
                str(empty_cell_token or "__EMPTY__")
                if normalize_empty_cells
                else ""
            ),
        }
    )

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    _write_complexity_artifacts(
        output_dir=output_dir,
        predictions=predictions,
        join_key=join_key,
        gt_field=gt_field,
        pred_field=pred_field,
        test_data=test_data,
        pred_data=pred_data,
    )

    print("\n" + "=" * 60)
    print("PRECOMPUTED EVALUATION RESULTS")
    print("=" * 60)
    print(f"  test_data:             {test_data}")
    print(f"  pred_data:             {pred_data}")
    print(f"  join_key:              {join_key}")
    print(f"  gt_field/pred_field:   {gt_field} vs {pred_field}")
    print(f"  normalize_empty_cells: {normalize_empty_cells}")
    if normalize_empty_cells:
        print(f"  empty_cell_token:      {empty_cell_token}")
    print(f"  test_total:            {len(test_rows)}")
    print(f"  matched_total:         {len(predictions)}")
    print(f"  unmatched_total:       {len(missing_rows)}")
    print(f"  pred_duplicate_keys:   {pred_duplicate_count}")
    print("  ---")
    print(f"  Avg TEDS:              {agg.avg_teds:.4f}")
    print(f"  Avg TEDS-Structure:    {agg.avg_teds_structure:.4f}")
    print(f"  Avg TEDS-Norm:         {agg.avg_teds_norm:.4f}")
    print(f"  Avg TEDS-Norm-S:       {agg.avg_teds_norm_structure:.4f}")
    print(f"  Avg Span F1:           {agg.avg_span_f1:.4f}")
    print(f"  Avg Attr Accuracy:     {agg.avg_attribute_accuracy:.4f}")
    print(f"  Avg Colspan Accuracy:  {agg.avg_colspan_accuracy:.4f}")
    print(f"  Avg Rowspan Accuracy:  {agg.avg_rowspan_accuracy:.4f}")
    print(f"  Avg GCA/GSA:           {agg.avg_gca:.4f} / {agg.avg_gsa:.4f}")
    print("  ---")
    print(f"  Simple/Medium/Complex: {agg.simple_teds:.4f} / {agg.medium_teds:.4f} / {agg.complex_teds:.4f}")
    print("=" * 60)
    print(f"  outputs: {output_dir}")
    print("=" * 60)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="사전 생성 JSONL 결과 평가")
    parser.add_argument("--test_data", required=True, help="테스트 기준 JSONL")
    parser.add_argument("--pred_data", required=True, help="비교 대상(JSONL)")
    parser.add_argument("--output_dir", required=True, help="출력 디렉토리")
    parser.add_argument("--join_key", default="image_path", help="조인 키")
    parser.add_argument("--gt_field", default="gt_html", help="테스트 JSONL GT 필드명")
    parser.add_argument("--pred_field", default="gt_html", help="비교 JSONL Pred 필드명")
    parser.add_argument("--complexity_field", default="complexity", help="복잡도 필드명")
    parser.add_argument("--max_samples", type=int, default=None, help="테스트 최대 샘플 수")
    parser.add_argument(
        "--strict_missing",
        action="store_true",
        help="조인 실패가 1건이라도 있으면 오류로 중단",
    )
    parser.add_argument(
        "--normalize_empty_cells",
        action="store_true",
        help="평가 전 GT/Pred HTML의 빈 셀을 empty_cell_token으로 정규화한다.",
    )
    parser.add_argument(
        "--empty_cell_token",
        default="__EMPTY__",
        help="빈 셀 표기에 사용할 토큰 (기본: __EMPTY__)",
    )
    args = parser.parse_args()

    evaluate_precomputed(
        test_data=args.test_data,
        pred_data=args.pred_data,
        output_dir=args.output_dir,
        join_key=args.join_key,
        gt_field=args.gt_field,
        pred_field=args.pred_field,
        complexity_field=args.complexity_field,
        max_samples=args.max_samples,
        strict_missing=args.strict_missing,
        normalize_empty_cells=args.normalize_empty_cells,
        empty_cell_token=args.empty_cell_token,
    )


if __name__ == "__main__":
    main()
