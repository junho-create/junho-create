"""
GT 품질 전수 조사 스크립트

predictions.jsonl을 기반으로 GT HTML의 구조적 품질을 분석한다.
- rowspan="2" 비율 계산 및 지그재그 패턴 탐지
- 행 수, 셀 수, colspan/rowspan 분포
- 저점수 샘플과 GT 품질 이슈 상관 분석
- 기존 gt_quality_filter 규칙 적용

Usage:
    python -m eval.analyze_gt_quality \
        --predictions eval_results/.../predictions.jsonl \
        --output eval_results/.../gt_quality_analysis.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.gt_quality_filter import validate_gt_html


def analyze_gt_html(html_str: str) -> dict:
    """GT HTML의 구조적 특성을 분석한다."""
    result = {
        "total_cells": 0,
        "total_rows": 0,
        "rowspan2_count": 0,
        "rowspan2_ratio": 0.0,
        "max_rowspan": 1,
        "max_colspan": 1,
        "colspan_cells": 0,
        "rowspan_cells": 0,
        "span_cells": 0,
        "empty_rows": 0,
        "is_zigzag_rowspan2": False,
        "gt_issues": [],
    }

    if not html_str or not html_str.strip():
        result["gt_issues"] = ["empty_html"]
        return result

    try:
        soup = BeautifulSoup(html_str, "html.parser")
    except Exception:
        result["gt_issues"] = ["parse_error"]
        return result

    table = soup.find("table")
    if not table:
        result["gt_issues"] = ["no_table_tag"]
        return result

    rows = table.find_all("tr")
    result["total_rows"] = len(rows)

    cells = table.find_all(["td", "th"])
    result["total_cells"] = len(cells)

    # 빈 행 카운트
    for row in rows:
        row_cells = row.find_all(["td", "th"])
        if len(row_cells) == 0:
            result["empty_rows"] += 1

    # 셀별 span 분석
    for cell in cells:
        rowspan = int(cell.get("rowspan", 1))
        colspan = int(cell.get("colspan", 1))

        if rowspan == 2:
            result["rowspan2_count"] += 1
        if rowspan > result["max_rowspan"]:
            result["max_rowspan"] = rowspan
        if colspan > result["max_colspan"]:
            result["max_colspan"] = colspan
        if rowspan > 1:
            result["rowspan_cells"] += 1
        if colspan > 1:
            result["colspan_cells"] += 1
        if rowspan > 1 or colspan > 1:
            result["span_cells"] += 1

    # rowspan=2 비율
    if result["total_cells"] > 0:
        result["rowspan2_ratio"] = result["rowspan2_count"] / result["total_cells"]

    # 지그재그 패턴 판별
    if result["total_cells"] >= 6 and result["rowspan2_ratio"] >= 0.5:
        result["is_zigzag_rowspan2"] = True

    # 기존 gt_quality_filter 규칙 적용
    result["gt_issues"] = validate_gt_html(html_str)

    return result


def run_analysis(predictions_path: str, output_path: str | None = None) -> dict:
    """predictions.jsonl 전수 조사를 수행한다."""
    # predictions 로드
    predictions = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    print(f"로드: {len(predictions)}개 predictions")

    # 각 샘플 분석
    analyses = []
    for pred in predictions:
        gt_html = pred.get("gt_html", "")
        gt_analysis = analyze_gt_html(gt_html)

        analyses.append({
            "image_path": pred.get("image_path", ""),
            "index": pred.get("index", -1),
            "complexity": pred.get("complexity", "unknown"),
            "teds": pred.get("teds", 0.0),
            "teds_structure": pred.get("teds_structure", 0.0),
            "teds_norm": pred.get("teds_norm", 0.0),
            "teds_norm_structure": pred.get("teds_norm_structure", 0.0),
            "span_f1": pred.get("span_f1", 0.0),
            **gt_analysis,
        })

    # === 통계 집계 ===
    total = len(analyses)

    # 지그재그 패턴 샘플
    zigzag_samples = [a for a in analyses if a["is_zigzag_rowspan2"]]

    # GT 이슈가 있는 샘플
    issue_samples = [a for a in analyses if a["gt_issues"]]

    # 이슈 카운터
    issue_counter = Counter()
    for a in analyses:
        for issue in a["gt_issues"]:
            # 동적 이슈명에서 접두사만 추출 (예: zigzag_rowspan2_75% → zigzag_rowspan2)
            prefix = issue.rsplit("_", 1)[0] if any(
                issue.startswith(p)
                for p in ("zigzag_rowspan2_", "excessive_colspan_", "excessive_rowspan_", "col_count_mismatch_")
            ) else issue
            issue_counter[prefix] += 1

    # 저점수 샘플 (TEDS-S <= 0.2)
    low_teds_s = [a for a in analyses if a["teds_structure"] <= 0.2]

    # rowspan2_ratio 분포
    ratio_buckets = {"0%": 0, "1-10%": 0, "10-30%": 0, "30-50%": 0, "50-80%": 0, "80-100%": 0}
    for a in analyses:
        r = a["rowspan2_ratio"]
        if r == 0:
            ratio_buckets["0%"] += 1
        elif r <= 0.1:
            ratio_buckets["1-10%"] += 1
        elif r <= 0.3:
            ratio_buckets["10-30%"] += 1
        elif r <= 0.5:
            ratio_buckets["30-50%"] += 1
        elif r <= 0.8:
            ratio_buckets["50-80%"] += 1
        else:
            ratio_buckets["80-100%"] += 1

    # 복잡도별 분포
    complexity_dist = Counter(a["complexity"] for a in analyses)
    zigzag_complexity = Counter(a["complexity"] for a in zigzag_samples)
    low_teds_complexity = Counter(a["complexity"] for a in low_teds_s)

    # TEDS vs TEDS-Norm 상승 폭 분석
    teds_improvements = []
    for a in analyses:
        delta = a["teds_norm"] - a["teds"]
        teds_improvements.append({
            "image_path": a["image_path"],
            "complexity": a["complexity"],
            "teds": a["teds"],
            "teds_norm": a["teds_norm"],
            "delta": delta,
            "is_zigzag": a["is_zigzag_rowspan2"],
        })

    # 상승 폭 top 20
    teds_improvements.sort(key=lambda x: x["delta"], reverse=True)
    top_improvements = teds_improvements[:20]

    # 상관 분석: 저점수 + GT 이슈
    low_with_issues = [a for a in low_teds_s if a["gt_issues"]]
    low_zigzag = [a for a in low_teds_s if a["is_zigzag_rowspan2"]]

    # 전체 평균
    avg_teds = sum(a["teds"] for a in analyses) / max(total, 1)
    avg_teds_norm = sum(a["teds_norm"] for a in analyses) / max(total, 1)
    avg_teds_s = sum(a["teds_structure"] for a in analyses) / max(total, 1)
    avg_teds_ns = sum(a["teds_norm_structure"] for a in analyses) / max(total, 1)

    # 결과 구성
    summary = {
        "total_samples": total,
        "complexity_distribution": dict(complexity_dist),
        "avg_teds": round(avg_teds, 4),
        "avg_teds_structure": round(avg_teds_s, 4),
        "avg_teds_norm": round(avg_teds_norm, 4),
        "avg_teds_norm_structure": round(avg_teds_ns, 4),
        "teds_norm_improvement": round(avg_teds_norm - avg_teds, 4),
        "zigzag_rowspan2": {
            "count": len(zigzag_samples),
            "ratio": round(len(zigzag_samples) / max(total, 1), 4),
            "complexity_distribution": dict(zigzag_complexity),
            "samples": [
                {
                    "image": os.path.basename(a["image_path"]),
                    "teds": round(a["teds"], 3),
                    "teds_norm": round(a["teds_norm"], 3),
                    "rowspan2_ratio": round(a["rowspan2_ratio"], 2),
                    "complexity": a["complexity"],
                }
                for a in sorted(zigzag_samples, key=lambda x: x["teds"])
            ],
        },
        "gt_issues": {
            "total_with_issues": len(issue_samples),
            "issue_counts": dict(issue_counter.most_common()),
        },
        "low_teds_s_analysis": {
            "count": len(low_teds_s),
            "complexity_distribution": dict(low_teds_complexity),
            "with_gt_issues": len(low_with_issues),
            "with_zigzag": len(low_zigzag),
            "gt_issue_ratio": round(
                len(low_with_issues) / max(len(low_teds_s), 1), 3
            ),
        },
        "rowspan2_ratio_distribution": ratio_buckets,
        "top_teds_norm_improvements": [
            {
                "image": os.path.basename(t["image_path"]),
                "teds": round(t["teds"], 3),
                "teds_norm": round(t["teds_norm"], 3),
                "delta": round(t["delta"], 3),
                "is_zigzag": t["is_zigzag"],
                "complexity": t["complexity"],
            }
            for t in top_improvements
        ],
    }

    # === 콘솔 출력 ===
    print("\n" + "=" * 70)
    print("GT QUALITY ANALYSIS REPORT")
    print("=" * 70)
    print(f"\n전체 샘플: {total}")
    print(f"복잡도 분포: {dict(complexity_dist)}")

    print(f"\n--- TEDS 평균 ---")
    print(f"  TEDS:       {avg_teds:.4f}")
    print(f"  TEDS-S:     {avg_teds_s:.4f}")
    print(f"  TEDS-N:     {avg_teds_norm:.4f}  (delta: {avg_teds_norm - avg_teds:+.4f})")
    print(f"  TEDS-NS:    {avg_teds_ns:.4f}  (delta: {avg_teds_ns - avg_teds_s:+.4f})")

    print(f"\n--- 지그재그 rowspan=2 ---")
    print(f"  탐지: {len(zigzag_samples)}개 ({len(zigzag_samples)/max(total,1)*100:.1f}%)")
    print(f"  복잡도: {dict(zigzag_complexity)}")
    for s in sorted(zigzag_samples, key=lambda x: x["teds"])[:10]:
        img = os.path.basename(s["image_path"])
        print(f"    {img:45s} TEDS={s['teds']:.3f} → N={s['teds_norm']:.3f}  rs2={s['rowspan2_ratio']:.0%}")

    print(f"\n--- GT 구조적 이슈 ---")
    print(f"  이슈 있는 샘플: {len(issue_samples)}개 ({len(issue_samples)/max(total,1)*100:.1f}%)")
    for issue, count in issue_counter.most_common(10):
        print(f"    {issue}: {count}개")

    print(f"\n--- 저점수 분석 (TEDS-S <= 0.2): {len(low_teds_s)}개 ---")
    print(f"  복잡도: {dict(low_teds_complexity)}")
    print(f"  GT 이슈 있음: {len(low_with_issues)}개 ({len(low_with_issues)/max(len(low_teds_s),1)*100:.1f}%)")
    print(f"  지그재그 rowspan2: {len(low_zigzag)}개 ({len(low_zigzag)/max(len(low_teds_s),1)*100:.1f}%)")

    print(f"\n--- rowspan=2 비율 분포 ---")
    for bucket, count in ratio_buckets.items():
        bar = "#" * (count // 5) if count > 0 else ""
        print(f"  {bucket:>10s}: {count:4d} {bar}")

    print(f"\n--- TEDS → TEDS-N 상승 Top 10 ---")
    for t in top_improvements[:10]:
        img = os.path.basename(t["image_path"])
        zigzag_mark = " [ZIGZAG]" if t["is_zigzag"] else ""
        print(f"  {img:45s} {t['teds']:.3f} → {t['teds_norm']:.3f} (delta={t['delta']:+.3f}){zigzag_mark}")

    print("=" * 70)

    # 결과 저장
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n저장: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="GT 품질 전수 조사")
    parser.add_argument(
        "--predictions",
        required=True,
        help="predictions.jsonl 경로",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="분석 결과 JSON 출력 경로 (미지정 시 predictions 옆에 생성)",
    )
    args = parser.parse_args()

    output_path = args.output
    if not output_path:
        pred_dir = os.path.dirname(args.predictions)
        output_path = os.path.join(pred_dir, "gt_quality_analysis.json")

    run_analysis(args.predictions, output_path)


if __name__ == "__main__":
    main()
