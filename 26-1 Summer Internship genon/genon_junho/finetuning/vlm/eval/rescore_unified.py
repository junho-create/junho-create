"""기존 통합 평가 결과(predictions_unified.jsonl)를 모델 추론 없이 재채점한다.

채점 로직(`eval/unified_metrics.py`)만 바뀌었고 모델 출력(`full_response`)은
그대로이므로, 재학습·재추론 없이 저장된 예측을 다시 점수화하기 위한 도구다.

핵심 동작
--------
- `full_response`(모델 원본 출력)는 predictions 파일에 이미 저장돼 있으므로 재사용.
- 정답 HTML/JSON 은:
  1) `--test_data` 가 주어지면 원본 test JSONL 의 `gt_html` 을 index 로 조인(table+layout 모두 정확).
  2) 없으면 table 은 predictions 에 저장된 `gt_table_html` 로 재채점하고,
     layout 은 기존 레코드 지표를 그대로 보존한다.

사용 예::

    python -m eval.rescore_unified \
        --predictions eval_results/unified_20260605_110341/predictions_unified.jsonl \
        --test_data data/.../unified_test.jsonl \
        --output_dir eval_results/unified_20260605_110341_rescored
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# eval 패키지 import 경로 보정 (스크립트 직접 실행 대비)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VLM_ROOT = os.path.dirname(_THIS_DIR)
if _VLM_ROOT not in sys.path:
    sys.path.insert(0, _VLM_ROOT)

from eval.unified_metrics import (  # noqa: E402
    aggregate_unified_metrics,
    evaluate_unified_sample,
)


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_gt_by_index(test_data_path: str) -> dict[int, dict]:
    """원본 test JSONL 을 line index 기준으로 매핑한다.

    evaluate_unified.py 의 _load_samples 는 index=enumerate 순서를 부여하므로
    동일한 line index 로 gt_html/task_type 을 복원할 수 있다.
    """
    mapping: dict[int, dict] = {}
    for i, rec in enumerate(_read_jsonl(test_data_path)):
        mapping[i] = {
            "gt_html": rec.get("gt_html", ""),
            "task_type": str(rec.get("task_type", "table")).strip().lower(),
        }
    return mapping


def rescore(
    predictions_path: str,
    output_dir: str,
    test_data_path: str | None = None,
    iou_threshold: float = 0.5,
) -> dict:
    preds = _read_jsonl(predictions_path)
    gt_map = _load_gt_by_index(test_data_path) if test_data_path else None

    new_records: list[dict] = []
    table_recovered = 0

    for p in preds:
        task = str(p.get("task_type", "table")).strip().lower()
        response = p.get("full_response", "")

        if gt_map is not None and p.get("index") in gt_map:
            gt = gt_map[p["index"]]
            gt_html = gt["gt_html"]
            task = gt["task_type"] or task
            rec = evaluate_unified_sample(response, gt_html, task, iou_threshold)
        elif task == "table":
            # test_data 없을 때: 저장된 gt_table_html 을 정답 JSON 으로 감싸 재채점
            gt_table_html = p.get("gt_table_html", "")
            gt_json = json.dumps(
                [{"bbox": [0, 0, 1024, 1024], "category": "Table", "text": gt_table_html}],
                ensure_ascii=False,
            )
            rec = evaluate_unified_sample(response, gt_json, "table", iou_threshold)
        else:
            # layout 은 gt 정보가 predictions 에 없으므로 기존 레코드 보존
            new_records.append(p)
            continue

        # 메타 필드 보존
        for k in ("index", "image_path", "prompt_style", "full_response", "inference_time"):
            if k in p:
                rec[k] = p[k]
        if rec.get("html_fallback_used", 0.0) >= 1.0:
            table_recovered += 1
        new_records.append(rec)

    agg = aggregate_unified_metrics(new_records)
    agg["rescored"] = True
    agg["rescored_from"] = os.path.abspath(predictions_path)
    agg["table_html_fallback_recovered"] = table_recovered
    if test_data_path:
        agg["rescore_gt_source"] = os.path.abspath(test_data_path)
    else:
        agg["rescore_gt_source"] = "predictions_gt_table_html (layout preserved)"

    os.makedirs(output_dir, exist_ok=True)
    pred_out = os.path.join(output_dir, "predictions_unified.jsonl")
    with open(pred_out, "w", encoding="utf-8") as f:
        for r in new_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    metrics_out = os.path.join(output_dir, "metrics_unified.json")
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)

    _print_summary(agg, table_recovered)
    print(f"\n  Saved: {metrics_out}")
    return agg


def _print_summary(agg: dict, table_recovered: int) -> None:
    print("\n" + "=" * 60)
    print("UNIFIED RE-SCORE RESULTS (no inference)")
    print("=" * 60)
    print(f"  Total samples:          {agg.get('total_samples', 0)}")
    print(f"  Table / Layout:         {agg.get('table_samples', 0)} / {agg.get('layout_samples', 0)}")
    print(f"  JSON parse success:     {agg.get('json_parse_success_rate', 0.0):.4f}")
    if "table" in agg:
        t = agg["table"]
        print("  --- Table ---")
        print(f"    Avg TEDS:             {t['avg_teds']:.4f}")
        print(f"    Avg TEDS-Structure:   {t['avg_teds_structure']:.4f}")
        print(f"    Avg Span F1:          {t['avg_span_f1']:.4f}")
        print(f"    JSON parse success:   {t['json_parse_success_rate']:.4f}")
        print(f"    HTML fallback rate:   {t.get('html_fallback_rate', 0.0):.4f}")
        print(f"    TEDS>0 rate:          {t.get('teds_nonzero_rate', 0.0):.4f}")
        print(f"    HTML-recovered count: {table_recovered}")
    if "layout" in agg:
        l = agg["layout"]
        print("  --- Layout ---")
        print(f"    Avg Element F1:       {l['avg_layout_f1']:.4f}")
    print("=" * 60)


def main() -> None:
    ap = argparse.ArgumentParser(description="통합 평가 결과 재채점(추론 없음)")
    ap.add_argument("--predictions", required=True, help="기존 predictions_unified.jsonl")
    ap.add_argument("--output_dir", required=True, help="재채점 결과 저장 디렉토리")
    ap.add_argument("--test_data", default=None, help="원본 통합 test JSONL (gt 조인용, 권장)")
    ap.add_argument("--iou_threshold", type=float, default=0.5)
    args = ap.parse_args()
    rescore(args.predictions, args.output_dir, args.test_data, args.iou_threshold)


if __name__ == "__main__":
    main()
