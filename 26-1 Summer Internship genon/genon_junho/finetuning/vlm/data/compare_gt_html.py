"""
GT HTML 비교 및 점수 구간별 분리

원본 GT와 Claude 생성 GT를 TEDS-S/TEDS로 비교하고,
TEDS-S 점수 구간에 따라 분리 저장한다.

Usage:
    python -m data.compare_gt_html \
        --original data/experiments/e7_gtfilter/train.jsonl \
        --claude data/gt_validation/train_claude.jsonl \
        --output_dir data/gt_validation/comparison/
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import TEDSCalculator


def _load_jsonl_by_image_path(path: str) -> dict[str, dict]:
    """JSONL을 image_path 키로 인덱싱하여 로드한다."""
    index = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ip = rec.get("image_path", "")
            if ip:
                index[ip] = rec
    return index


def _classify(teds_s: float, threshold_high: float, threshold_med: float) -> str:
    """TEDS-S 점수에 따른 구간 분류."""
    if teds_s >= 1.0 - 1e-9:
        return "perfect"
    elif teds_s >= threshold_high:
        return "high"
    elif teds_s >= threshold_med:
        return "medium"
    else:
        return "low"


def compare_gt_html(
    original_path: str,
    claude_path: str,
    output_dir: str,
    threshold_high: float = 0.8,
    threshold_med: float = 0.5,
    split_count: int = 100,
) -> dict:
    """원본 GT와 Claude GT를 비교하고 점수 구간별 분리 저장한다.

    Returns:
        요약 통계 dict
    """
    print(f"원본 로드: {original_path}")
    original_index = _load_jsonl_by_image_path(original_path)
    print(f"  → {len(original_index)}건")

    print(f"Claude GT 로드: {claude_path}")
    claude_index = _load_jsonl_by_image_path(claude_path)
    print(f"  → {len(claude_index)}건")

    # 매칭
    matched_keys = set(original_index.keys()) & set(claude_index.keys())
    unmatched_original = set(original_index.keys()) - set(claude_index.keys())
    unmatched_claude = set(claude_index.keys()) - set(original_index.keys())

    print(f"매칭: {len(matched_keys)}건, 원본만: {len(unmatched_original)}건, Claude만: {len(unmatched_claude)}건")

    # TEDS 계산기
    teds_s_calc = TEDSCalculator(structure_only=True)

    # 출력 준비
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list[dict]] = {
        "perfect": [],
        "high": [],
        "medium": [],
        "low": [],
        "error": [],
    }

    # 복잡도별 통계
    complexity_stats: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"teds_s": [], "teds": []}
    )

    all_teds_s = []

    for ip in tqdm(sorted(matched_keys), desc="TEDS 비교"):
        orig_rec = original_index[ip]
        claude_rec = claude_index[ip]

        orig_gt = orig_rec.get("gt_html", "")
        claude_gt = claude_rec.get("gt_html", "")

        # Claude 생성 실패 (빈값 또는 에러)
        if not claude_gt or claude_rec.get("error"):
            out_rec = {
                **claude_rec,
                "teds_s_score": 0.0,
                "error": claude_rec.get("error", "empty_response"),
            }
            buckets["error"].append(out_rec)
            continue

        # TEDS 계산
        teds_s = teds_s_calc.compute(claude_gt, orig_gt)
        all_teds_s.append(teds_s)

        # 복잡도 통계
        complexity = claude_rec.get("complexity", "unknown")
        complexity_stats[complexity]["teds_s"].append(teds_s)

        # 출력 레코드: Claude 결과
        out_rec = {
            **claude_rec,
            "teds_s_score": round(teds_s, 6),
        }

        bucket = _classify(teds_s, threshold_high, threshold_med)
        buckets[bucket].append(out_rec)

    # 구간별 JSONL 저장
    for bucket_name, recs in buckets.items():
        # perfect가 아니면 split_count 건 단위로 나누어 저장
        if bucket_name != "perfect":
            for i in range(0, len(recs), split_count):
                out_path = out_dir / f"{bucket_name}_{i // split_count + 1}.jsonl"
                with open(out_path, "w", encoding="utf-8") as f:
                    for rec in recs[i:i + split_count]:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            out_path = out_dir / f"{bucket_name}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in recs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 요약 통계 생성
    def _avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    summary = {
        "total_original": len(original_index),
        "total_claude": len(claude_index),
        "matched": len(matched_keys),
        "unmatched_original": len(unmatched_original),
        "unmatched_claude": len(unmatched_claude),
        "thresholds": {
            "perfect": "== 1.0",
            "high": f">= {threshold_high}",
            "medium": f">= {threshold_med}",
            "low": f"< {threshold_med}",
        },
        "buckets": {name: len(recs) for name, recs in buckets.items()},
        "overall": {
            "avg_teds_s": round(_avg(all_teds_s), 6),
            "count": len(all_teds_s),
        },
        "by_complexity": {},
    }

    for comp in sorted(complexity_stats.keys()):
        stats = complexity_stats[comp]
        summary["by_complexity"][comp] = {
            "count": len(stats["teds_s"]),
            "avg_teds_s": round(_avg(stats["teds_s"]), 6),
        }

    # summary.json 저장
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 결과 출력
    print("\n" + "=" * 60)
    print("GT 비교 결과")
    print("=" * 60)
    print(f"  전체 비교: {summary['overall']['count']}건")
    print(f"  평균 TEDS-S: {summary['overall']['avg_teds_s']:.4f}")
    print(f"  ---")
    for name in ("perfect", "high", "medium", "low", "error"):
        count = summary["buckets"][name]
        pct = count / max(len(matched_keys), 1) * 100
        print(f"  {name:>8s}: {count:>6d}건 ({pct:5.1f}%)")
    print(f"  ---")
    print(f"  복잡도별:")
    for comp, stats in summary["by_complexity"].items():
        print(f"    {comp}: {stats['count']}건, TEDS-S={stats['avg_teds_s']:.4f}")
    print(f"  ---")
    print(f"  출력 디렉토리: {output_dir}")
    print(f"  summary: {summary_path}")
    print("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="GT HTML 비교 및 점수 구간별 분리")
    parser.add_argument("--original", required=True, help="원본 slim JSONL")
    parser.add_argument("--claude", required=True, help="Claude 생성 JSONL")
    parser.add_argument("--output_dir", required=True, help="출력 디렉토리")
    parser.add_argument("--threshold_high", type=float, default=0.8, help="high 구간 하한")
    parser.add_argument("--threshold_med", type=float, default=0.5, help="medium 구간 하한")
    parser.add_argument("--split_count", type=int, default=100, help="perfect이 아닌 경우 몇 건마다 파일을 나눌지")
    args = parser.parse_args()

    for path in (args.original, args.claude):
        if not Path(path).exists():
            print(f"Error: 파일을 찾을 수 없습니다: {path}", file=sys.stderr)
            sys.exit(1)

    compare_gt_html(
        original_path=args.original,
        claude_path=args.claude,
        output_dir=args.output_dir,
        threshold_high=args.threshold_high,
        threshold_med=args.threshold_med,
        split_count=args.split_count,
    )


if __name__ == "__main__":
    main()
