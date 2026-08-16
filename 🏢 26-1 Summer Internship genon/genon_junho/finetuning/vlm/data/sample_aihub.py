"""
인덱스 기반 복잡도 비율 샘플링

analyze_aihub.py가 생성한 인덱스 JSONL을 기반으로
복잡도 비율에 따라 샘플링하여 학습용 JSONL을 생성한다.

Usage:
    python -m data.sample_aihub \
        --index ./data/index/training_index.jsonl \
        --output ./data/experiments/e7/train_raw.jsonl \
        --count 10000 \
        --ratio_complex 0.30 \
        --ratio_medium 0.40 \
        --ratio_simple 0.30 \
        --prompt_style chandra_table_without_ocr \
        --hard_first \
        --seed 42 \
        --exclude path1.jsonl path2.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.convert_aihub import clean_aihub_html, extract_table_html  # noqa: E402
from utils.html_utils import extract_spans_from_html, normalize_html, parse_html_table  # noqa: E402
from utils.prompt_templates import (  # noqa: E402
    build_thinking_chain,
    normalize_prompt_style,
)


def _load_index(index_path: str) -> list[dict]:
    """인덱스 JSONL을 로드한다."""
    entries = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _extract_filename(path_like: str) -> str:
    """경로 문자열에서 파일명만 추출한다."""
    if not path_like:
        return ""
    cleaned = path_like.strip().replace("\\", "/")
    if not cleaned:
        return ""
    # URL 쿼리/프래그먼트가 섞여 들어온 경우 방어
    cleaned = cleaned.split("?", 1)[0].split("#", 1)[0]
    return cleaned.rsplit("/", 1)[-1]


def _load_exclude_filenames(exclude_paths: list[str]) -> set[str]:
    """--exclude JSONL 파일들에서 image_path를 추출해 파일명 기준 제외 목록을 구성한다."""
    excluded = set()
    for path in exclude_paths:
        p = Path(path)
        if not p.exists():
            print(f"  경고: exclude 파일 없음, 건너뜀: {path}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                image_path = record.get("image_path", "")
                filename = _extract_filename(image_path)
                if filename:
                    excluded.add(filename)
    return excluded


def _ratio_targets(total: int, ratio_complex: float, ratio_medium: float, ratio_simple: float) -> dict[str, int]:
    """비율에 따라 복잡도별 할당량을 계산한다 (largest-remainder method)."""
    ratio_sum = ratio_complex + ratio_medium + ratio_simple
    if ratio_sum <= 0:
        ratio_complex = ratio_medium = ratio_simple = 1.0 / 3
        ratio_sum = 1.0

    normalized = {
        "complex": ratio_complex / ratio_sum,
        "medium": ratio_medium / ratio_sum,
        "simple": ratio_simple / ratio_sum,
    }

    floors = {}
    remainders = []
    for comp, ratio in normalized.items():
        exact = total * ratio
        base = int(math.floor(exact))
        floors[comp] = base
        remainders.append((exact - base, comp))

    remaining = total - sum(floors.values())
    for _, comp in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        floors[comp] += 1
        remaining -= 1

    return floors


def _parse_ratio_args(args_list: Optional[list[str]]) -> dict[str, float]:
    """'cat=ratio' 형식의 리스트를 dict로 파싱한다."""
    if not args_list:
        return {}
    result = {}
    for item in args_list:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        try:
            result[k.strip()] = float(v.strip())
        except ValueError:
            pass
    return result


def _get_complexity_label(entry: dict, use_detail: bool) -> str:
    """use_detail 활성화 시 complex → complex_nested/col/row/mix 세분화 레이블 반환."""
    comp = entry.get("complexity", "simple")
    if not use_detail or comp != "complex":
        return comp
    if entry.get("nested_table_count", 0) > 0:
        return "complex_nested"
    pat = entry.get("span_pattern", "none")
    mapping = {"col_only": "complex_col", "row_only": "complex_row", "mixed": "complex_mix"}
    return mapping.get(pat, "complex_mix")


def _equalize_by_attr(
    pool: list[dict],
    attr_key: str,
    target_ratios: dict[str, float],
    total: int,
    rng: random.Random,
) -> list[dict]:
    """특정 속성(attr_key) 기준으로 pool을 서브풀로 나눠 목표 비율에 맞게 추출한다."""
    sub_pools: dict[str, list] = {}
    for entry in pool:
        cat = entry.get(attr_key, "unknown")
        sub_pools.setdefault(cat, []).append(entry)

    result = []
    for cat, ratio in target_ratios.items():
        n = int(total * ratio)
        sub = sub_pools.get(cat, [])
        rng.shuffle(sub)
        result.extend(sub[:n])

    # 부족분 backfill
    selected_set = set(id(e) for e in result)
    leftover = [e for e in pool if id(e) not in selected_set]
    remaining = total - len(result)
    if remaining > 0:
        rng.shuffle(leftover)
        result.extend(leftover[:remaining])
    return result


def _build_thinking(normalized_html: str, num_rows: int, num_cols: int, has_header: bool) -> str:
    """Thinking chain을 생성한다."""
    spans = extract_spans_from_html(normalized_html)
    structure = parse_html_table(normalized_html)
    header_rows = max(structure.header_rows, 1 if has_header else 0)
    return build_thinking_chain(
        num_rows=num_rows,
        num_cols=num_cols,
        spans=spans,
        header_rows=header_rows,
    )


def _build_record(entry: dict, prompt_style: str, no_thinking: bool, use_detail: bool = False) -> dict:
    """인덱스 엔트리에서 slim 포맷 학습용 레코드를 생성한다.

    Slim 포맷:
        image_path, gt_html, thinking, complexity, prompt_style
        (ocr_info, bbox_scale는 add_ocr.py에서 별도 추가)
    """
    image_path = entry["image_path"]
    normalized_html = entry["normalized_html"]
    num_rows = entry["num_rows"]
    num_cols = entry["num_cols"]
    has_header = entry.get("has_header", False)

    thinking = ""
    if not no_thinking:
        thinking = _build_thinking(normalized_html, num_rows, num_cols, has_header)

    return {
        "image_path": image_path,
        "gt_html": normalized_html,
        "thinking": thinking,
        "complexity": _get_complexity_label(entry, use_detail),
        "prompt_style": prompt_style,
    }


def run(args: argparse.Namespace) -> None:
    started = time.time()
    prompt_style = normalize_prompt_style(args.prompt_style)
    if args.prompt_style != prompt_style:
        print(f"경고: '{args.prompt_style}' -> '{prompt_style}'")

    # 1. 인덱스 로드
    print(f"인덱스 로드: {args.index}")
    entries = _load_index(args.index)
    print(f"  총 엔트리: {len(entries)}")

    # 2. file_id 기준 중복 제거
    seen_file_ids = set()
    unique_entries = []
    for entry in entries:
        fid = entry.get("file_id", "")
        if fid and fid in seen_file_ids:
            continue
        if fid:
            seen_file_ids.add(fid)
        unique_entries.append(entry)
    dedup_count = len(entries) - len(unique_entries)
    if dedup_count > 0:
        print(f"  file_id 중복 제거: {dedup_count}건")
    entries = unique_entries

    # 3. exclude 처리
    if args.exclude:
        exclude_filenames = _load_exclude_filenames(args.exclude)
        before = len(entries)
        entries = [
            e for e in entries
            if _extract_filename(e.get("image_path", "")) not in exclude_filenames
        ]
        excluded = before - len(entries)
        print(
            f"  exclude 적용(파일명 기준): {excluded}건 제외 "
            f"(exclude filenames={len(exclude_filenames)}, 남은: {len(entries)})"
        )

    # 4. 구조 유사도 기반 중복제거
    #    동일한 structure_signature를 가진 샘플을 max_per_signature개로 제한한다.
    #    구조가 유사한 샘플이 과도하게 포함되면 모델이 특정 패턴에 과적합될 수 있다.
    if args.max_per_signature > 0:
        signature_counts: Counter = Counter()
        filtered_entries = []
        similarity_excluded = 0
        for entry in entries:
            sig = entry.get("structure_signature", "")
            if sig and signature_counts[sig] >= args.max_per_signature:
                similarity_excluded += 1
                continue
            if sig:
                signature_counts[sig] += 1
            filtered_entries.append(entry)
        if similarity_excluded > 0:
            print(f"  구조 유사도 필터: {similarity_excluded}건 제외 "
                  f"(max_per_signature={args.max_per_signature}, 남은: {len(filtered_entries)})")
        entries = filtered_entries

    # 4-2. GT 구조적 품질 필터
    gt_excluded = 0
    if args.gt_quality_filter:
        from data.gt_quality_filter import validate_gt_html

        before = len(entries)
        filtered = []
        gt_issues_counter: Counter = Counter()
        for entry in entries:
            html = entry.get("normalized_html", "")
            issues = validate_gt_html(html)
            if issues:
                for issue in issues:
                    gt_issues_counter[issue] += 1
            else:
                filtered.append(entry)
        gt_excluded = before - len(filtered)
        entries = filtered
        print(f"  GT 품질 필터: {gt_excluded}건 제외 (남은: {len(entries)})")
        if gt_issues_counter:
            for issue, cnt in gt_issues_counter.most_common(5):
                print(f"    {issue}: {cnt}건")

    if len(entries) < args.count:
        raise ValueError(
            f"샘플 수 부족: 요청={args.count}, 가용={len(entries)}"
        )

    # 5. 복잡도별 풀 구성
    rng = random.Random(args.seed)

    if args.use_complex_detail:
        # complex를 4개 서브버킷으로 세분화
        pools: dict[str, list] = {
            "complex_nested": [], "complex_col": [], "complex_row": [],
            "complex_mix": [], "medium": [], "simple": [],
        }
        for entry in entries:
            comp = entry.get("complexity", "simple")
            if comp == "complex":
                # 2차 기준 필터 적용
                if args.filter_table_size and entry.get("table_size_cat") not in args.filter_table_size:
                    continue
                if args.filter_max_span and entry.get("max_span_cat") not in args.filter_max_span:
                    continue
                if args.filter_grid_irregularity and entry.get("grid_irregularity_cat") not in args.filter_grid_irregularity:
                    continue
                # 1차 분류: 중첩 테이블 우선, 이후 span_pattern
                if entry.get("nested_table_count", 0) > 0:
                    bucket = "complex_nested"
                else:
                    pat = entry.get("span_pattern", "none")
                    pat_map = {"col_only": "complex_col", "row_only": "complex_row", "mixed": "complex_mix"}
                    bucket = pat_map.get(pat, "complex_mix")
                pools[bucket].append(entry)
            elif comp in ("medium", "simple"):
                pools[comp].append(entry)

        print(f"  복잡도 풀(세분화): "
              f"complex_nested={len(pools['complex_nested'])}, "
              f"complex_col={len(pools['complex_col'])}, "
              f"complex_row={len(pools['complex_row'])}, "
              f"complex_mix={len(pools['complex_mix'])}, "
              f"medium={len(pools['medium'])}, "
              f"simple={len(pools['simple'])}")

        # 비율 할당량 계산 (세분화 모드)
        ratio_sum = (args.ratio_complex_nested + args.ratio_complex_col +
                     args.ratio_complex_row + args.ratio_complex_mix +
                     args.ratio_medium + args.ratio_simple)
        if ratio_sum <= 0:
            ratio_sum = 1.0

        def _n(ratio: float) -> int:
            return max(0, round(args.count * ratio / ratio_sum))

        targets = {
            "complex_nested": _n(args.ratio_complex_nested),
            "complex_col":    _n(args.ratio_complex_col),
            "complex_row":    _n(args.ratio_complex_row),
            "complex_mix":    _n(args.ratio_complex_mix),
            "medium":         _n(args.ratio_medium),
            "simple":         _n(args.ratio_simple),
        }
        # 반올림 오차 보정
        diff = args.count - sum(targets.values())
        if diff != 0:
            targets["medium"] = max(0, targets["medium"] + diff)

        print(f"  할당 목표(세분화): {targets}")
        pool_order = ("complex_nested", "complex_col", "complex_row", "complex_mix", "medium", "simple")

    else:
        # 기존 3-pool 모드 (하위 호환)
        pools = {"complex": [], "medium": [], "simple": []}
        for entry in entries:
            comp = entry.get("complexity", "simple")
            if comp not in pools:
                comp = "simple"
            pools[comp].append(entry)

        print(f"  복잡도 풀: complex={len(pools['complex'])}, "
              f"medium={len(pools['medium'])}, simple={len(pools['simple'])}")

        # 2차 지표 기반 서브풀 균등화 (use_complex_detail 미사용 시에만 적용)
        target_table_size = _parse_ratio_args(args.target_table_size)
        target_max_span = _parse_ratio_args(args.target_max_span)
        target_grid_irregularity = _parse_ratio_args(args.target_grid_irregularity)

        complex_target = round(args.count * args.ratio_complex /
                               max(args.ratio_complex + args.ratio_medium + args.ratio_simple, 1e-9))
        if target_table_size:
            pools["complex"] = _equalize_by_attr(pools["complex"], "table_size_cat", target_table_size, complex_target, rng)
        elif target_max_span:
            pools["complex"] = _equalize_by_attr(pools["complex"], "max_span_cat", target_max_span, complex_target, rng)
        elif target_grid_irregularity:
            pools["complex"] = _equalize_by_attr(pools["complex"], "grid_irregularity_cat", target_grid_irregularity, complex_target, rng)

        targets = _ratio_targets(args.count, args.ratio_complex, args.ratio_medium, args.ratio_simple)
        print(f"  할당 목표: {targets}")
        pool_order = ("complex", "medium", "simple")

    # 7. 각 풀 정렬/섞기
    for comp in pool_order:
        if args.hard_first:
            # grid_irregularity → max_span → table_size → complexity_score 높은 순
            pools[comp].sort(
                key=lambda x: (
                    x.get("grid_irregularity", 0.0),
                    x.get("max_span", 1),
                    x.get("table_size_cells", 0),
                    x.get("complexity_score", 0.0),
                ),
                reverse=True,
            )
        else:
            rng.shuffle(pools[comp])

    # 8. 복잡도별 선택
    selected = []
    for comp in pool_order:
        target = targets[comp]
        available = pools[comp][:target]
        selected.extend(available)
        pools[comp] = pools[comp][target:]

    # 9. 부족분 backfill
    remaining = args.count - len(selected)
    if remaining > 0:
        for comp in pool_order:
            if remaining <= 0:
                break
            backfill = pools[comp][:remaining]
            selected.extend(backfill)
            pools[comp] = pools[comp][len(backfill):]
            remaining -= len(backfill)

    if len(selected) < args.count:
        raise ValueError(
            f"샘플 수 충족 실패: 요청={args.count}, 선택={len(selected)}"
        )

    # 최종 셔플
    rng.shuffle(selected)

    # 10. 레코드 빌드
    print(f"레코드 빌드 중... (prompt_style={prompt_style})")
    records = []
    build_failed = 0
    for entry in tqdm(selected, desc="레코드 빌드"):
        try:
            record = _build_record(entry, prompt_style, args.no_thinking, args.use_complex_detail)
            records.append(record)
        except Exception:
            build_failed += 1

    # 11. 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 12. 리포트
    complexity_dist = Counter(r["complexity"] for r in records)
    elapsed = round(time.time() - started, 2)

    # 지표별 분포 집계 (complex 계열만 집계)
    selected_index = {e["image_path"]: e for e in selected}
    complex_image_paths = [
        r["image_path"]
        for r in records
        if str(r.get("complexity", "")).startswith("complex")
    ]

    def _count_dist(attr_key: str) -> dict:
        c: Counter = Counter()
        for img_path in complex_image_paths:
            cat = selected_index.get(img_path, {}).get(attr_key, "unknown")
            c[cat] += 1
        return dict(c)

    report = {
        "requested_count": args.count,
        "final_count": len(records),
        "build_failed": build_failed,
        "complexity_distribution": dict(complexity_dist),
        "ratio_targets": targets,
        "seed": args.seed,
        "hard_first": args.hard_first,
        "max_per_signature": args.max_per_signature,
        "gt_quality_excluded": gt_excluded,
        "prompt_style": prompt_style,
        "elapsed_sec": elapsed,
        "use_complex_detail": args.use_complex_detail,
        "complex_metrics_count": len(complex_image_paths),
        # 5개 지표별 분포
        "span_pattern_distribution":       _count_dist("span_pattern"),
        "table_size_distribution":         _count_dist("table_size_cat"),
        "max_span_distribution":           _count_dist("max_span_cat"),
        "grid_irregularity_distribution":  _count_dist("grid_irregularity_cat"),
        "nested_table_distribution":       _count_dist("nested_table_cat"),
    }

    report_path = str(output_path.with_suffix(".report.json"))
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n=== 샘플링 완료 ===")
    print(f"  요청:    {args.count}")
    print(f"  결과:    {len(records)}")
    print(f"  빌드실패: {build_failed}")
    print(f"  복잡도:  {dict(complexity_dist)}")
    print(f"  소요:    {elapsed}s")
    print(f"  출력:    {output_path}")
    print(f"  리포트:  {report_path}")


def main():
    parser = argparse.ArgumentParser(description="인덱스 기반 복잡도 비율 샘플링")
    parser.add_argument(
        "--index",
        required=True,
        help="analyze_aihub.py가 생성한 인덱스 JSONL 경로",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="출력 JSONL 경로",
    )
    parser.add_argument("--count", type=int, required=True, help="샘플 수")
    parser.add_argument("--ratio_complex", type=float, default=0.30, help="complex 비율 (use_complex_detail 미사용 시)")
    parser.add_argument("--ratio_medium", type=float, default=0.40, help="medium 비율")
    parser.add_argument("--ratio_simple", type=float, default=0.30, help="simple 비율")
    # ── 1차 분류 세분화 (use_complex_detail 활성화 시) ─────────────────────────
    parser.add_argument(
        "--use_complex_detail",
        action="store_true",
        help="complex를 complex_nested/complex_col/complex_row/complex_mix로 세분화",
    )
    parser.add_argument("--ratio_complex_nested", type=float, default=0.03, help="complex_nested 비율")
    parser.add_argument("--ratio_complex_col",    type=float, default=0.10, help="complex_col 비율")
    parser.add_argument("--ratio_complex_row",    type=float, default=0.05, help="complex_row 비율")
    parser.add_argument("--ratio_complex_mix",    type=float, default=0.12, help="complex_mix 비율")
    # ── 2차 지표 필터 (use_complex_detail 활성화 시, complex 풀 필터링) ──────────
    parser.add_argument(
        "--filter_table_size",
        nargs="*",
        metavar="CAT",
        help="complex 내 table_size_cat 필터 (예: small medium large)",
    )
    parser.add_argument(
        "--filter_max_span",
        nargs="*",
        metavar="CAT",
        help="complex 내 max_span_cat 필터 (예: normal large very_large)",
    )
    parser.add_argument(
        "--filter_grid_irregularity",
        nargs="*",
        metavar="CAT",
        help="complex 내 grid_irregularity_cat 필터 (예: regular moderate irregular)",
    )
    # ── 2차 지표 목표 비율 (use_complex_detail 미사용 시, complex 단일 풀 균등화) ─
    parser.add_argument(
        "--target_table_size",
        nargs="*",
        metavar="cat=ratio",
        help="complex 내 table_size 목표 비율 (예: small=0.2 medium=0.5 large=0.3)",
    )
    parser.add_argument(
        "--target_max_span",
        nargs="*",
        metavar="cat=ratio",
        help="complex 내 max_span 목표 비율 (예: normal=0.5 large=0.3 very_large=0.2)",
    )
    parser.add_argument(
        "--target_grid_irregularity",
        nargs="*",
        metavar="cat=ratio",
        help="complex 내 grid_irregularity 목표 비율 (예: regular=0.3 moderate=0.4 irregular=0.3)",
    )
    parser.add_argument(
        "--prompt_style",
        default="chandra_table_without_ocr",
        help="프롬프트 스타일 (기본: chandra_table_without_ocr)",
    )
    parser.add_argument(
        "--hard_first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="어려운 샘플 우선 (complexity_score 높은 순, 기본: true)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="제외할 JSONL 파일 경로들 (이미 사용된 파일명 기준 제외)",
    )
    parser.add_argument(
        "--max_per_signature",
        type=int,
        default=5,
        help="동일 구조 서명당 최대 샘플 수 (0=비활성화, 기본: 5)",
    )
    parser.add_argument(
        "--no_thinking",
        action="store_true",
        help="thinking chain 생성 비활성화",
    )
    parser.add_argument(
        "--gt_quality_filter",
        action="store_true",
        help="GT HTML 구조적 품질 필터 활성화 (빈 행, 비정상 span 등 제외)",
    )
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count는 양수여야 합니다")
    for name in ("ratio_complex", "ratio_medium", "ratio_simple"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name}는 0 이상이어야 합니다")

    run(args)


if __name__ == "__main__":
    main()
