"""
GT Source 통일 재계산 스크립트

기존 predictions.jsonl의 pred_html을 보존하고,
GT만 aihub 원본 HTML로 교체하여 모든 메트릭을 재계산한다.

문제 배경:
  - 서빙 서버 4모델은 gt_source=dataset (JSONL의 gt_html 사용)
  - 학습 서버 3모델은 gt_source=aihub (원본 .html 파일 사용)
  - 동일 모델(e1init2000)의 +0.1744 TEDS 차이는 주로 max_model_len 차이이나,
    GT 통일 없이는 공정한 비교 불가
  - 또한 구 코드의 simple_teds/medium_teds/complex_teds에 TEDS-S 혼입 버그 수정

기능:
  1. predictions.jsonl에서 pred_html, image_path_raw 로드
  2. image_path_raw → .html 파일 경로 → aihub GT HTML 로드
  3. TEDSCalculator로 teds, teds_structure, teds_norm, teds_norm_structure 재계산
  4. compute_span_metrics(), compute_grid_metrics() 재계산
  5. 새 predictions.jsonl + metrics.json + 복잡도별 파일 저장
  6. compute_aggregate_metrics() 사용 (simple_teds 버그 수정된 현재 코드)

Usage:
    python -m eval.recalculate_metrics \
        --predictions_dirs eval_results/model_a eval_results/model_b \
        --data_root /home/vlm_train/qwen3_vl_tsr \
        --output_dir eval_results/20260225_unified_aihub_gt

    # 또는 단일 디렉토리
    python -m eval.recalculate_metrics \
        --predictions_dirs eval_results/model_a \
        --data_root . \
        --output_dir eval_results/model_a_unified
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

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


# =========================================================================
# GT 로드 유틸리티 (evaluate.py에서 핵심 로직만 추출)
# =========================================================================


def _extract_table_from_full_html(full_html: str) -> str:
    """전체 HTML 문서에서 <table>...</table> 조각을 추출한다."""
    if not full_html:
        return ""
    match = re.search(r"(<table[\s\S]*?</table>)", full_html, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _load_aihub_gt_html_from_path(image_path: str) -> tuple[str, str]:
    """이미지 경로에서 대응하는 aihub GT HTML을 로드한다.

    Returns:
        (table_html, html_file_path)  — 실패 시 ("", "")
    """
    html_path = str(Path(image_path).with_suffix(".html"))
    if not os.path.exists(html_path):
        return "", ""

    try:
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            full_html = f.read()
    except Exception:
        return "", ""

    table_html = _extract_table_from_full_html(full_html)
    if not table_html:
        return "", html_path

    # 기본값 속성(span=1) 제거로 GT 비교 잡음 축소
    table_html = table_html.replace(' rowspan="1"', "")
    table_html = table_html.replace(' colspan="1"', "")
    return table_html, html_path


def _resolve_image_path(image_path_raw: str, data_root: str) -> str:
    """image_path_raw를 data_root 기준으로 해석해 실제 경로를 반환한다."""
    if not image_path_raw:
        return ""

    p = Path(image_path_raw)

    # 1) 절대 경로이고 존재하면 그대로 사용
    if p.is_absolute() and p.exists():
        return str(p)

    # 2) data_root 기준 상대 경로
    candidate = Path(data_root) / p
    if candidate.exists():
        return str(candidate.resolve())

    # 3) data_root를 한 단계 위로 올려서 시도
    candidate2 = Path(data_root).parent / p
    if candidate2.exists():
        return str(candidate2.resolve())

    # 4) 현재 작업 디렉토리 기준
    candidate3 = Path.cwd() / p
    if candidate3.exists():
        return str(candidate3.resolve())

    return ""


# =========================================================================
# 핵심 재계산 로직
# =========================================================================


def recalculate_single_dir(
    predictions_dir: str,
    data_root: str,
    output_dir: str,
    normalize_empty_cells: bool = False,
    empty_cell_token: str = "__EMPTY__",
) -> dict:
    """단일 predictions 디렉토리를 재계산한다.

    Returns:
        요약 dict: {model_name, total, aihub_loaded, aihub_failed, metrics}
    """
    pred_path = os.path.join(predictions_dir, "predictions.jsonl")
    if not os.path.exists(pred_path):
        print(f"  [SKIP] predictions.jsonl 없음: {predictions_dir}")
        return {}

    # 기존 metrics.json에서 메타 정보 읽기
    old_metrics_path = os.path.join(predictions_dir, "metrics.json")
    old_metrics = {}
    if os.path.exists(old_metrics_path):
        with open(old_metrics_path, "r", encoding="utf-8") as f:
            old_metrics = json.load(f)

    model_name = Path(predictions_dir).name
    print(f"\n{'='*60}")
    print(f"  재계산: {model_name}")
    print(f"  소스: {pred_path}")
    print(f"  기존 gt_source: {old_metrics.get('gt_source', 'unknown')}")
    print(f"{'='*60}")

    # predictions.jsonl 로드
    records = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"  총 레코드: {len(records)}")

    # TEDS 계산기 초기화
    teds_calc = TEDSCalculator(structure_only=False)
    teds_struct_calc = TEDSCalculator(structure_only=True)
    teds_norm_calc = TEDSCalculator(structure_only=False, normalize=True)
    teds_norm_struct_calc = TEDSCalculator(structure_only=True, normalize=True)

    # 재계산
    new_predictions = []
    aihub_loaded = 0
    aihub_missing = 0
    aihub_parse_failed = 0

    start_time = time.time()

    for i, record in enumerate(records):
        pred_html = record.get("pred_html", "")
        image_path_raw = record.get("image_path_raw", "")
        complexity = record.get("complexity", "unknown")
        old_gt_source = record.get("gt_source", "dataset")

        # aihub GT 로드 시도
        resolved_path = _resolve_image_path(image_path_raw, data_root)
        gt_html = ""
        gt_html_path = ""
        gt_source_used = "dataset"

        if resolved_path:
            aihub_gt, html_path = _load_aihub_gt_html_from_path(resolved_path)
            if aihub_gt:
                gt_html = aihub_gt
                gt_source_used = "aihub"
                gt_html_path = html_path
                aihub_loaded += 1
            elif html_path:
                aihub_parse_failed += 1
                # 파싱 실패 시 기존 gt_html 유지
                gt_html = record.get("gt_html", "")
                gt_html_path = html_path
            else:
                aihub_missing += 1
                gt_html = record.get("gt_html", "")
        else:
            aihub_missing += 1
            gt_html = record.get("gt_html", "")

        pred_html_eval = _normalize_empty_cells_html(
            pred_html,
            enabled=normalize_empty_cells,
            empty_cell_token=empty_cell_token,
        )
        gt_html_eval = _normalize_empty_cells_html(
            gt_html,
            enabled=normalize_empty_cells,
            empty_cell_token=empty_cell_token,
        )

        # 메트릭 재계산
        teds = teds_calc.compute(pred_html_eval, gt_html_eval)
        teds_struct = teds_struct_calc.compute(pred_html_eval, gt_html_eval)
        teds_norm = teds_norm_calc.compute(pred_html_eval, gt_html_eval)
        teds_norm_struct = teds_norm_struct_calc.compute(pred_html_eval, gt_html_eval)
        span_metrics = compute_span_metrics(pred_html_eval, gt_html_eval)
        grid_metrics = compute_grid_metrics(pred_html_eval, gt_html_eval)

        new_record = {
            "index": record.get("index", i),
            "image_path": record.get("image_path", ""),
            "image_path_raw": image_path_raw,
            "pred_html": pred_html,
            "pred_html_eval": pred_html_eval,
            "gt_html": gt_html,
            "gt_html_eval": gt_html_eval,
            "gt_source": gt_source_used,
            "gt_html_path": gt_html_path,
            "full_response": record.get("full_response", ""),
            "complexity": complexity,
            "teds": teds,
            "teds_structure": teds_struct,
            "teds_norm": teds_norm,
            "teds_norm_structure": teds_norm_struct,
            "span_f1": span_metrics.span_f1,
            "span_precision": span_metrics.span_precision,
            "span_recall": span_metrics.span_recall,
            "position_f1": span_metrics.position_f1,
            "attribute_accuracy": span_metrics.attribute_accuracy,
            "colspan_accuracy": span_metrics.colspan_accuracy,
            "rowspan_accuracy": span_metrics.rowspan_accuracy,
            "gca": grid_metrics.gca,
            "gsa": grid_metrics.gsa,
            "inference_time": record.get("inference_time", 0.0),
        }
        new_predictions.append(new_record)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"    진행: {i+1}/{len(records)} ({elapsed:.1f}s)")

    elapsed_total = time.time() - start_time
    print(f"  완료: {len(new_predictions)}개 ({elapsed_total:.1f}s)")
    print(
        f"  GT 소스: aihub_loaded={aihub_loaded}, "
        f"aihub_missing={aihub_missing}, aihub_parse_failed={aihub_parse_failed}"
    )

    # 출력 디렉토리 생성
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # predictions.jsonl 저장
    pred_out_path = os.path.join(model_output_dir, "predictions.jsonl")
    with open(pred_out_path, "w", encoding="utf-8") as f:
        for p in new_predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  저장: {pred_out_path}")

    # aggregate metrics 계산 (현재 코드의 compute_aggregate_metrics 사용)
    agg = compute_aggregate_metrics(new_predictions)

    # metrics.json 저장
    metrics_dict = agg.to_dict()
    total_inf_time = sum(
        float(p.get("inference_time", 0.0) or 0.0) for p in new_predictions
    )
    metrics_dict["avg_inference_time"] = total_inf_time / max(len(new_predictions), 1)
    metrics_dict["backend"] = old_metrics.get("backend", "unknown")
    metrics_dict["batch_size"] = old_metrics.get("batch_size", 0)
    metrics_dict["max_new_tokens"] = old_metrics.get("max_new_tokens", 10000)
    metrics_dict["gt_source"] = "aihub"
    metrics_dict["strict_aihub_gt"] = False
    metrics_dict["recalculated_from"] = predictions_dir
    metrics_dict["recalculated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metrics_dict["aihub_loaded"] = aihub_loaded
    metrics_dict["aihub_missing"] = aihub_missing
    metrics_dict["aihub_parse_failed"] = aihub_parse_failed
    metrics_dict["normalize_empty_cells"] = bool(normalize_empty_cells)
    metrics_dict["empty_cell_token"] = (
        str(empty_cell_token or "__EMPTY__")
        if normalize_empty_cells
        else ""
    )

    metrics_path = os.path.join(model_output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    print(f"  저장: {metrics_path}")

    # 복잡도별 분리 저장 (base + detail 동시 지원)
    base_order = ["simple", "medium", "complex"]
    detail_order = ["complex_nested", "complex_col", "complex_row", "complex_mix"]
    by_base = {k: [] for k in base_order}
    by_raw: dict[str, list[dict]] = {}
    for pred in new_predictions:
        raw_comp = normalize_complexity_label(pred.get("complexity", "unknown"))
        by_raw.setdefault(raw_comp, []).append(pred)

        base_comp = to_base_complexity_bucket(raw_comp)
        if base_comp in by_base:
            by_base[base_comp].append(pred)

    ordered_raw = [c for c in detail_order if c in by_raw]
    extra_raw = sorted(
        c for c in by_raw.keys()
        if c not in set(base_order) and c not in set(detail_order)
    )
    ordered_raw.extend(extra_raw)

    artifact_sets: list[tuple[str, list[dict]]] = []
    for comp in base_order:
        artifact_sets.append((comp, by_base[comp]))
    for comp in ordered_raw:
        artifact_sets.append((comp, by_raw[comp]))

    for comp, subset in artifact_sets:
        # predictions_<complexity>.jsonl
        comp_pred_path = os.path.join(model_output_dir, f"predictions_{comp}.jsonl")
        with open(comp_pred_path, "w", encoding="utf-8") as f:
            for p in subset:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        # metrics_<complexity>.json
        if subset:
            comp_agg = compute_aggregate_metrics(subset)
            comp_metrics = comp_agg.to_dict()
            comp_inf_time = sum(
                float(p.get("inference_time", 0.0) or 0.0) for p in subset
            )
            comp_metrics["avg_inference_time"] = comp_inf_time / max(len(subset), 1)
            comp_metrics["gt_source"] = "aihub"
            comp_metrics["normalize_empty_cells"] = bool(normalize_empty_cells)
            comp_metrics["empty_cell_token"] = (
                str(empty_cell_token or "__EMPTY__")
                if normalize_empty_cells
                else ""
            )
        else:
            comp_metrics = {
                "total_samples": 0,
                "gt_source": "aihub",
                "normalize_empty_cells": bool(normalize_empty_cells),
                "empty_cell_token": (
                    str(empty_cell_token or "__EMPTY__")
                    if normalize_empty_cells
                    else ""
                ),
            }

        comp_metrics_path = os.path.join(model_output_dir, f"metrics_{comp}.json")
        with open(comp_metrics_path, "w", encoding="utf-8") as f:
            json.dump(comp_metrics, f, indent=2, ensure_ascii=False)

    # 결과 요약 출력
    print(f"\n  --- 재계산 결과 ---")
    print(f"  Avg TEDS:           {agg.avg_teds:.4f}")
    print(f"  Avg TEDS-S:         {agg.avg_teds_structure:.4f}")
    print(f"  Avg TEDS-Norm:      {agg.avg_teds_norm:.4f}")
    print(f"  Avg TEDS-Norm-S:    {agg.avg_teds_norm_structure:.4f}")
    print(f"  Avg Span F1:        {agg.avg_span_f1:.4f}")
    print(f"  Avg Attr Acc:       {agg.avg_attribute_accuracy:.4f}")
    print(f"  Avg Colspan Acc:    {agg.avg_colspan_accuracy:.4f}")
    print(f"  Avg Rowspan Acc:    {agg.avg_rowspan_accuracy:.4f}")
    print(f"  Avg GCA:            {agg.avg_gca:.4f}")
    print(f"  Avg GSA:            {agg.avg_gsa:.4f}")
    print(f"  Simple TEDS:        {agg.simple_teds:.4f}")
    print(f"  Medium TEDS:        {agg.medium_teds:.4f}")
    print(f"  Complex TEDS:       {agg.complex_teds:.4f}")

    return {
        "model_name": model_name,
        "total": len(new_predictions),
        "aihub_loaded": aihub_loaded,
        "aihub_missing": aihub_missing,
        "aihub_parse_failed": aihub_parse_failed,
        "old_gt_source": old_metrics.get("gt_source", "unknown"),
        "old_avg_teds": old_metrics.get("avg_teds", 0.0),
        "new_avg_teds": agg.avg_teds,
        "delta_teds": agg.avg_teds - old_metrics.get("avg_teds", 0.0),
        "metrics": metrics_dict,
    }


def print_comparison_table(results: list[dict]) -> None:
    """재계산 전후 비교 테이블을 출력한다."""
    print("\n" + "=" * 80)
    print("GT Source 통일 재계산 비교 요약")
    print("=" * 80)

    header = (
        f"{'Model':<45} {'Old GT':>8} {'Old TEDS':>10} {'New TEDS':>10} "
        f"{'ΔTEDS':>8} {'AiHub':>6}"
    )
    print(header)
    print("-" * 80)

    for r in results:
        if not r:
            continue
        name = r["model_name"][:44]
        old_gt = r["old_gt_source"]
        old_teds = r["old_avg_teds"]
        new_teds = r["new_avg_teds"]
        delta = r["delta_teds"]
        aihub = r["aihub_loaded"]
        sign = "+" if delta >= 0 else ""
        print(
            f"{name:<45} {old_gt:>8} {old_teds:>10.4f} {new_teds:>10.4f} "
            f"{sign}{delta:>7.4f} {aihub:>5}/{r['total']}"
        )

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="GT Source 통일 재계산: 기존 predictions의 pred_html을 보존하고 "
        "GT를 aihub 원본으로 교체하여 메트릭 재계산"
    )
    parser.add_argument(
        "--predictions_dirs",
        nargs="+",
        required=True,
        help="재계산할 predictions.jsonl이 있는 디렉토리 (복수 가능)",
    )
    parser.add_argument(
        "--data_root",
        default=".",
        help="AI-Hub 데이터 루트 경로 (image_path_raw 해석 기준). "
        "예: /home/vlm_train/qwen3_vl_tsr",
    )
    parser.add_argument(
        "--output_dir",
        default="eval_results/unified_aihub_gt",
        help="재계산 결과 저장 디렉토리",
    )
    parser.add_argument(
        "--normalize_empty_cells",
        action="store_true",
        help="재계산 전 GT/Pred HTML의 빈 셀을 empty_cell_token으로 정규화한다.",
    )
    parser.add_argument(
        "--empty_cell_token",
        default="__EMPTY__",
        help="빈 셀 표기에 사용할 토큰 (기본: __EMPTY__)",
    )
    args = parser.parse_args()

    print(f"데이터 루트: {args.data_root}")
    print(f"출력 디렉토리: {args.output_dir}")
    print(f"빈 셀 정규화: {args.normalize_empty_cells}")
    if args.normalize_empty_cells:
        print(f"빈 셀 토큰: {args.empty_cell_token}")
    print(f"대상 디렉토리: {len(args.predictions_dirs)}개")

    results = []
    for pred_dir in args.predictions_dirs:
        result = recalculate_single_dir(
            predictions_dir=pred_dir,
            data_root=args.data_root,
            output_dir=args.output_dir,
            normalize_empty_cells=args.normalize_empty_cells,
            empty_cell_token=args.empty_cell_token,
        )
        results.append(result)

    # 전체 비교 요약
    print_comparison_table(results)

    # 요약 JSON 저장
    summary_path = os.path.join(args.output_dir, "recalculation_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {
        "recalculated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_root": args.data_root,
        "gt_source": "aihub",
        "normalize_empty_cells": bool(args.normalize_empty_cells),
        "empty_cell_token": (
            str(args.empty_cell_token or "__EMPTY__")
            if args.normalize_empty_cells
            else ""
        ),
        "models": [],
    }
    for r in results:
        if not r:
            continue
        summary["models"].append(
            {
                "model_name": r["model_name"],
                "total_samples": r["total"],
                "old_gt_source": r["old_gt_source"],
                "old_avg_teds": round(r["old_avg_teds"], 4),
                "new_avg_teds": round(r["new_avg_teds"], 4),
                "delta_teds": round(r["delta_teds"], 4),
                "aihub_loaded": r["aihub_loaded"],
                "aihub_missing": r["aihub_missing"],
                "aihub_parse_failed": r["aihub_parse_failed"],
            }
        )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n요약 저장: {summary_path}")


if __name__ == "__main__":
    main()
