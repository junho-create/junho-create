"""
GT 품질 필터링 - 교차 모델 검증 + 구조적 검증

여러 모델의 평가 predictions를 교차 분석하여 GT 의심 샘플을 식별한다.
- 교차 모델 검증: max(N models TEDS) < threshold → suspect
- 구조적 검증: 빈 행, 셀 수 불일치, 비정상 span 등

출력:
  - suspect_samples.jsonl: sample_aihub.py --exclude 호환 포맷
  - clean_test.jsonl: 의심 제외 평가 데이터 (옵션)
  - 콘솔 요약 리포트

Usage:
    python -m data.gt_quality_filter \
        --predictions_dirs eval_results/dir1 eval_results/dir2 ... \
        --test_data data/.../test.jsonl \
        --threshold 0.5 \
        --output_dir data/gt_quality/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from bs4 import BeautifulSoup


def load_predictions(pred_dir: str) -> dict[str, dict]:
    """predictions.jsonl 로드 → {file_id: record}"""
    pred_file = os.path.join(pred_dir, "predictions.jsonl")
    if not os.path.exists(pred_file):
        print(f"  경고: {pred_file} 없음, 건너뜀")
        return {}

    records = {}
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # file_id 추출 (image_path에서 파일명)
            img_path = rec.get("image_path", "")
            file_id = os.path.splitext(os.path.basename(img_path))[0]
            records[file_id] = rec
    return records


def validate_gt_html(html_str: str) -> list[str]:
    """GT HTML의 구조적 문제를 검사. 발견된 이슈 목록 반환."""
    issues = []

    if not html_str or not html_str.strip():
        issues.append("empty_html")
        return issues

    # 빈 <tr></tr> 행 존재
    if re.search(r"<tr[^>]*>\s*</tr>", html_str, re.IGNORECASE):
        issues.append("empty_tr_row")

    try:
        soup = BeautifulSoup(html_str, "html.parser")
    except Exception:
        issues.append("parse_error")
        return issues

    table = soup.find("table")
    if not table:
        issues.append("no_table_tag")
        return issues

    rows = table.find_all("tr")
    if len(rows) == 0:
        issues.append("zero_rows")
        return issues

    # 각 행의 셀 수 + span 검사
    max_cols = 0
    for row in rows:
        cells = row.find_all(["td", "th"])
        col_count = 0
        for cell in cells:
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))
            col_count += colspan
            # 비정상 span 값
            if colspan > 20:
                issues.append(f"excessive_colspan_{colspan}")
            if rowspan > 50:
                issues.append(f"excessive_rowspan_{rowspan}")
        if col_count > max_cols:
            max_cols = col_count

    # 행 간 컬럼 수 불일치 (rough check)
    col_counts = []
    for row in rows:
        cells = row.find_all(["td", "th"])
        col_count = sum(int(c.get("colspan", 1)) for c in cells)
        col_counts.append(col_count)

    if col_counts and max(col_counts) > 0:
        # rowspan이 있으면 행별 col 수가 다를 수 있으므로, 극단적 차이만 체크
        min_cols = min(c for c in col_counts if c > 0) if any(c > 0 for c in col_counts) else 0
        if min_cols > 0 and max_cols > min_cols * 3:
            issues.append(f"col_count_mismatch_{min_cols}_vs_{max_cols}")

    # --- 지그재그 rowspan=2 패턴 검출 ---
    # 모든 td/th 셀에서 rowspan="2" 비율 계산
    all_cells = table.find_all(["td", "th"])
    total_cells = len(all_cells)
    if total_cells >= 6:  # 최소 셀 수 조건
        rowspan2_count = sum(
            1 for c in all_cells
            if int(c.get("rowspan", 1)) == 2
        )
        ratio = rowspan2_count / total_cells
        if ratio >= 0.5:
            issues.append(f"zigzag_rowspan2_{ratio:.0%}")

    return issues


def cross_validate_models(
    model_predictions: dict[str, dict[str, dict]],
    threshold: float = 0.5,
    mild_threshold: float = 0.6,
) -> list[dict]:
    """
    교차 모델 검증: 모든 모델의 TEDS를 수집하여 suspect 판별.

    Returns: [{"file_id": ..., "suspect_level": ..., "max_teds": ..., ...}, ...]
    """
    # 모든 file_id 수집
    all_file_ids = set()
    for model_name, preds in model_predictions.items():
        all_file_ids.update(preds.keys())

    results = []
    for file_id in sorted(all_file_ids):
        model_scores = {}
        gt_html = None
        image_path = None
        complexity = None

        for model_name, preds in model_predictions.items():
            if file_id in preds:
                rec = preds[file_id]
                model_scores[model_name] = rec.get("teds", 0.0)
                if gt_html is None:
                    gt_html = rec.get("gt_html", "")
                if image_path is None:
                    image_path = rec.get("image_path", "")
                if complexity is None:
                    complexity = rec.get("complexity", "unknown")

        if not model_scores:
            continue

        max_teds = max(model_scores.values())
        mean_teds = sum(model_scores.values()) / len(model_scores)

        reasons = []

        # 교차 모델 검증
        if max_teds < threshold:
            suspect_level = "strong"
            reasons.append(f"cross_model_low(max={max_teds:.3f}<{threshold})")
        elif max_teds < mild_threshold:
            suspect_level = "mild"
            reasons.append(f"cross_model_marginal(max={max_teds:.3f}<{mild_threshold})")
        else:
            suspect_level = "clean"

        # 구조적 GT 검증
        gt_issues = validate_gt_html(gt_html) if gt_html else []
        if gt_issues:
            reasons.extend([f"structural:{i}" for i in gt_issues])

        results.append({
            "file_id": file_id,
            "image_path": image_path,
            "complexity": complexity,
            "suspect_level": suspect_level,
            "max_teds": round(max_teds, 4),
            "mean_teds": round(mean_teds, 4),
            "model_scores": {k: round(v, 4) for k, v in model_scores.items()},
            "reasons": reasons,
            "gt_issues": gt_issues,
        })

    return results


def generate_report(results: list[dict], model_names: list[str]):
    """콘솔 요약 리포트 출력."""
    total = len(results)
    strong = [r for r in results if r["suspect_level"] == "strong"]
    mild = [r for r in results if r["suspect_level"] == "mild"]
    clean = [r for r in results if r["suspect_level"] == "clean"]

    print("=" * 70)
    print("GT QUALITY FILTER REPORT")
    print("=" * 70)
    print(f"\n전체 샘플: {total}")
    print(f"  strong suspect (max TEDS < threshold): {len(strong)} ({len(strong)/max(total,1)*100:.1f}%)")
    print(f"  mild suspect   (max TEDS < mild_th):   {len(mild)} ({len(mild)/max(total,1)*100:.1f}%)")
    print(f"  clean:                                  {len(clean)} ({len(clean)/max(total,1)*100:.1f}%)")

    # 복잡도별 분포
    print(f"\n--- 복잡도별 suspect 분포 ---")
    for level_name, level_list in [("strong", strong), ("mild", mild)]:
        comp_dist = Counter(r["complexity"] for r in level_list)
        print(f"  {level_name}: {dict(comp_dist)}")

    # 구조적 이슈 통계
    all_issues = Counter()
    for r in results:
        for issue in r["gt_issues"]:
            all_issues[issue] += 1
    if all_issues:
        print(f"\n--- 구조적 GT 이슈 ---")
        for issue, count in all_issues.most_common(10):
            print(f"  {issue}: {count}개")

    # TEDS < 0.3인 extreme 샘플 상위 20개
    extreme = sorted([r for r in results if r["max_teds"] < 0.3], key=lambda x: x["max_teds"])
    if extreme:
        print(f"\n--- Extreme (max TEDS < 0.3): {len(extreme)}개 ---")
        for r in extreme[:20]:
            scores_str = " ".join(f"{k}={v:.3f}" for k, v in r["model_scores"].items())
            print(f"  {r['file_id']:45s} {r['complexity']:8s} max={r['max_teds']:.3f} | {scores_str}")

    # strong suspect 전체 목록 (TEDS 순)
    print(f"\n--- Strong Suspect 전체 ({len(strong)}개, max TEDS 순) ---")
    for r in sorted(strong, key=lambda x: x["max_teds"]):
        reasons_short = ", ".join(r["reasons"][:2])
        print(f"  {r['file_id']:45s} {r['complexity']:8s} max={r['max_teds']:.3f} mean={r['mean_teds']:.3f} | {reasons_short}")

    print("=" * 70)


def save_suspect_jsonl(
    results: list[dict],
    output_path: str,
    include_levels: list[str] = None,
):
    """
    suspect 샘플을 JSONL로 저장. sample_aihub.py --exclude 호환 포맷.
    """
    if include_levels is None:
        include_levels = ["strong"]

    suspects = [r for r in results if r["suspect_level"] in include_levels]
    with open(output_path, "w", encoding="utf-8") as f:
        for r in sorted(suspects, key=lambda x: x["max_teds"]):
            # sample_aihub.py --exclude 호환: metadata.image_path 필수
            record = {
                "metadata": {
                    "image_path": r["image_path"],
                    "file_id": r["file_id"],
                },
                "suspect_level": r["suspect_level"],
                "max_teds": r["max_teds"],
                "mean_teds": r["mean_teds"],
                "model_scores": r["model_scores"],
                "reasons": r["reasons"],
                "complexity": r["complexity"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n저장: {output_path} ({len(suspects)}개 suspect)")


def save_clean_test_jsonl(
    results: list[dict],
    test_data_path: str,
    output_path: str,
    include_levels: list[str] = None,
):
    """
    suspect 제외한 clean test.jsonl 생성.
    """
    if include_levels is None:
        include_levels = ["strong"]

    suspect_ids = {
        r["file_id"]
        for r in results
        if r["suspect_level"] in include_levels
    }

    kept = 0
    excluded = 0
    with open(test_data_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # file_id 추출
            meta = rec.get("metadata", {})
            file_id = meta.get("file_id", "") or meta.get("sample_id", "")
            if not file_id:
                img_path = meta.get("image_path", "")
                file_id = os.path.splitext(os.path.basename(img_path))[0]

            if file_id in suspect_ids:
                excluded += 1
            else:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1

    print(f"저장: {output_path} (kept={kept}, excluded={excluded})")


def main():
    parser = argparse.ArgumentParser(description="GT 품질 필터링 (교차 모델 검증)")
    parser.add_argument(
        "--predictions_dirs",
        nargs="+",
        required=True,
        help="predictions.jsonl이 있는 디렉토리들 (모델별)",
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=None,
        help="모델 이름 (predictions_dirs와 1:1 매핑). 미지정 시 디렉토리명에서 추출",
    )
    parser.add_argument(
        "--test_data",
        default=None,
        help="원본 test.jsonl (clean_test.jsonl 생성용)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="strong suspect 판별 threshold (기본 0.5)",
    )
    parser.add_argument(
        "--mild_threshold",
        type=float,
        default=0.6,
        help="mild suspect 판별 threshold (기본 0.6)",
    )
    parser.add_argument(
        "--output_dir",
        default="data/gt_quality",
        help="출력 디렉토리",
    )
    parser.add_argument(
        "--include_mild",
        action="store_true",
        help="mild suspect도 제외 대상에 포함",
    )
    args = parser.parse_args()

    # 모델별 predictions 로드
    model_predictions = {}
    model_names_list = []

    for i, pred_dir in enumerate(args.predictions_dirs):
        if args.model_names and i < len(args.model_names):
            name = args.model_names[i]
        else:
            # 디렉토리명에서 모델명 추출
            dirname = os.path.basename(pred_dir.rstrip("/"))
            # serving_api_XXX_prompt... 패턴에서 모델명 추출
            name = dirname
            for prefix in ["serving_api_", "local_api_"]:
                if name.startswith(prefix):
                    name = name[len(prefix):]
            # _promptocr 이전까지 추출
            for suffix in ["_promptocr", "_prompt_noocr", "_prompt_"]:
                idx = name.find(suffix)
                if idx > 0:
                    name = name[:idx]
                    break

        print(f"로딩: {name} ← {pred_dir}")
        preds = load_predictions(pred_dir)
        if preds:
            model_predictions[name] = preds
            model_names_list.append(name)
            print(f"  → {len(preds)}개 predictions")

    if not model_predictions:
        print("오류: 유효한 predictions가 없습니다.")
        sys.exit(1)

    # 교차 검증
    print(f"\n교차 검증 수행 (models={len(model_predictions)}, threshold={args.threshold}, mild={args.mild_threshold})")
    results = cross_validate_models(
        model_predictions,
        threshold=args.threshold,
        mild_threshold=args.mild_threshold,
    )

    # 리포트 출력
    generate_report(results, model_names_list)

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # suspect JSONL 저장
    include_levels = ["strong"]
    if args.include_mild:
        include_levels.append("mild")

    suspect_path = os.path.join(args.output_dir, "suspect_samples.jsonl")
    save_suspect_jsonl(results, suspect_path, include_levels=include_levels)

    # 전체 결과 저장 (분석용)
    all_results_path = os.path.join(args.output_dir, "all_samples_quality.jsonl")
    with open(all_results_path, "w", encoding="utf-8") as f:
        for r in sorted(results, key=lambda x: x["max_teds"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"저장: {all_results_path} (전체 {len(results)}개)")

    # clean test.jsonl 생성 (test_data 지정 시)
    if args.test_data:
        clean_path = os.path.join(args.output_dir, "clean_test.jsonl")
        save_clean_test_jsonl(results, args.test_data, clean_path, include_levels=include_levels)


if __name__ == "__main__":
    main()
