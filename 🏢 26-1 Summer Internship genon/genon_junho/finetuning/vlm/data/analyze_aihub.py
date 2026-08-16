"""
AIHub 전체 데이터 분석 + 인덱싱

모든 이미지-HTML 쌍을 파싱하여 복잡도 분류 결과를 인덱스 JSONL로 저장한다.
이후 sample_aihub.py에서 이 인덱스를 기반으로 샘플링한다.

Usage:
    python -m data.analyze_aihub \
        --input_dir ./data/extracted/aihub/Training \
        --output ./data/index/training_index.jsonl \
        --skip        # output 파일 존재 시 재분석 생략 (기본값)
        --no_skip     # 강제 재분석
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.convert_aihub import (  # noqa: E402
    clean_aihub_html,
    extract_table_html,
    find_pairs_from_directory,
    load_aihub_metadata,
)
from utils.html_utils import normalize_html, parse_html_table  # noqa: E402
from utils.span_analyzer import (  # noqa: E402
    analyze_span,
    table_size_category,
    max_span_category,
    grid_irregularity_category,
)


# AIHub 표 유형 정규화
SPAN_TABLE_TYPES = {"조합표", "병합표", "콘텐츠 병합표"}


def canonical_table_type(raw_type: Optional[str]) -> str:
    """AIHub 표 유형 레이블을 정규화한다."""
    t = str(raw_type or "").strip()
    k = t.replace(" ", "")
    if not k:
        return "미상"
    if "기본표" in k:
        return "기본표"
    if "콘텐츠" in k and "병합" in k:
        return "콘텐츠 병합표"
    if "조합표" in k:
        return "조합표"
    if "병합표" in k:
        return "병합표"
    return t


def _pick_meta_value(meta: dict, table_meta: dict, keys: tuple[str, ...], default=None):
    """meta/table_meta 양쪽에서 첫 유효 값을 가져온다."""
    for key in keys:
        if isinstance(table_meta, dict):
            v = table_meta.get(key)
            if v not in (None, ""):
                return v
        if isinstance(meta, dict):
            v = meta.get(key)
            if v not in (None, ""):
                return v
    return default


def _extract_span_tokens(table_html: str) -> tuple[str, int, int, int]:
    """
    <td>/<th> 태그에서 구조 토큰 시퀀스를 생성한다.

    동일한 행·열 수와 span 패턴을 가진 테이블은 같은 시퀀스를 갖게 되어
    구조 유사도 판별에 사용된다.

    Returns:
        (span_token_string, tr_count, cell_count, th_count)
    """
    html = table_html.lower()
    tr_count = len(re.findall(r"<tr\b", html))
    th_count = len(re.findall(r"<th\b", html))

    tags = re.findall(r"<t[dh]\b[^>]*>", html)
    cell_count = len(tags)
    tokens: list[str] = []
    for tag in tags:
        rs_match = re.search(r'rowspan="(\d+)"', tag)
        cs_match = re.search(r'colspan="(\d+)"', tag)
        rs = int(rs_match.group(1)) if rs_match else 1
        cs = int(cs_match.group(1)) if cs_match else 1
        cell_type = "h" if tag.startswith("<th") else "d"
        tokens.append(f"{cell_type}:{rs}x{cs}")

    return ",".join(tokens), tr_count, cell_count, th_count


def build_structure_signature(
    normalized_html: str,
    num_rows: int,
    num_cols: int,
    num_span_cells: int,
) -> str:
    """
    테이블 구조의 해시 서명을 생성한다.

    행·열 수, span 셀 수, 셀별 span 토큰 시퀀스를 결합하여
    구조적으로 동일/유사한 테이블에 같은 서명을 부여한다.
    """
    span_tokens, tr_count, cell_count, th_count = _extract_span_tokens(normalized_html)
    signature_raw = (
        f"rows={num_rows}|cols={num_cols}|span_cells={num_span_cells}|"
        f"tr={tr_count}|cells={cell_count}|th={th_count}|spans={span_tokens}"
    )
    return hashlib.sha1(signature_raw.encode("utf-8")).hexdigest()


def analyze_pair(pair: dict) -> Optional[dict]:
    """단일 이미지-HTML 쌍을 분석하여 인덱스 엔트리를 반환한다."""
    html_path = pair.get("html_path")
    if not html_path:
        return None

    try:
        with open(html_path, "r", encoding="utf-8") as f:
            full_html = f.read()
    except Exception:
        return None

    table_html = extract_table_html(full_html)
    if not table_html:
        return None

    try:
        cleaned_html = clean_aihub_html(table_html)
        normalized = normalize_html(cleaned_html)
        structure = parse_html_table(normalized)
        if structure.num_rows == 0 or structure.num_cols == 0:
            return None
        span_stats = analyze_span(normalized)
    except Exception:
        return None

    # 메타데이터 로드
    meta = None
    if pair.get("json_path"):
        meta = load_aihub_metadata(pair["json_path"])

    table_meta = meta.get("table_meta", {}) if isinstance(meta, dict) else {}

    table_type_raw = _pick_meta_value(
        meta=meta if isinstance(meta, dict) else {},
        table_meta=table_meta if isinstance(table_meta, dict) else {},
        keys=("table_meta.table_type", "table_type"),
        default="",
    )
    table_field_raw = _pick_meta_value(
        meta=meta if isinstance(meta, dict) else {},
        table_meta=table_meta if isinstance(table_meta, dict) else {},
        keys=("table_meta.table_field", "table_field"),
        default="",
    )
    table_header_raw = _pick_meta_value(
        meta=meta if isinstance(meta, dict) else {},
        table_meta=table_meta if isinstance(table_meta, dict) else {},
        keys=("table_meta.table_header", "table_header"),
        default="",
    )

    signature = build_structure_signature(
        normalized_html=normalized,
        num_rows=structure.num_rows,
        num_cols=structure.num_cols,
        num_span_cells=span_stats.span_cells,
    )

    return {
        "file_id": pair.get("file_id", ""),
        "image_path": pair["image_path"],
        "html_path": html_path,
        "json_path": pair.get("json_path"),
        "normalized_html": normalized,
        "complexity": span_stats.complexity,
        "complexity_score": round(span_stats.complexity_score, 4),
        "num_rows": structure.num_rows,
        "num_cols": structure.num_cols,
        "num_span_cells": span_stats.span_cells,
        "structure_signature": signature,
        "table_type": canonical_table_type(table_type_raw),
        "table_field": str(table_field_raw or ""),
        "has_header": str(table_header_raw or "").upper() == "Y",
        # 5개 세분화 지표 신규 필드
        "span_pattern":          span_stats.span_pattern,
        "table_size_cells":      span_stats.table_size_cells,
        "table_size_cat":        table_size_category(span_stats.table_size_cells),
        "max_span":              span_stats.max_span,
        "max_span_cat":          max_span_category(span_stats.max_span),
        "grid_irregularity":     span_stats.grid_irregularity,
        "grid_irregularity_cat": grid_irregularity_category(span_stats.grid_irregularity),
        "nested_table_count":    span_stats.nested_table_count,
        "nested_table_cat":      "nested" if span_stats.nested_table_count > 0 else "none",
    }


def run(args: argparse.Namespace) -> None:
    output_path = Path(args.output)

    # --skip: 이미 인덱스 파일이 있으면 건너뛰기
    if args.skip and output_path.exists() and output_path.stat().st_size > 0:
        # 기존 인덱스 건수 확인
        count = sum(1 for _ in open(output_path, "r", encoding="utf-8") if _.strip())
        print(f"인덱스 파일 존재, 분석 생략: {output_path} ({count}건)")
        return

    print(f"데이터 디렉토리 탐색: {args.input_dir}")
    pairs = find_pairs_from_directory(args.input_dir)
    print(f"  발견된 쌍: {len(pairs)}")

    if not pairs:
        raise ValueError("쌍을 찾지 못했습니다. input_dir 경로를 확인하세요.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    complexity_counter = Counter()

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in tqdm(pairs, desc="HTML 분석 + 인덱싱"):
            entry = analyze_pair(pair)
            if entry is None:
                stats["parse_failed"] += 1
                continue

            stats["parse_ok"] += 1
            complexity_counter[entry["complexity"]] += 1
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 통계 요약
    total = stats["parse_ok"] + stats["parse_failed"]
    print(f"\n=== 분석 완료 ===")
    print(f"  총 쌍:      {total}")
    print(f"  성공:        {stats['parse_ok']}")
    print(f"  실패:        {stats['parse_failed']}")
    print(f"  복잡도 분포: {dict(complexity_counter)}")
    print(f"  출력:        {output_path}")


def main():
    parser = argparse.ArgumentParser(description="AIHub 전체 데이터 분석 + 인덱싱")
    parser.add_argument(
        "--input_dir",
        default="./data/extracted/aihub/Training",
        help="압축 해제된 AIHub 디렉토리 (01.원천데이터, 02.라벨링데이터 포함)",
    )
    parser.add_argument(
        "--output",
        default="./data/index/training_index.jsonl",
        help="출력 인덱스 JSONL 경로",
    )
    parser.add_argument(
        "--skip",
        action="store_true",
        default=True,
        help="output 파일 존재 시 재분석 생략 (기본값)",
    )
    parser.add_argument(
        "--no_skip",
        action="store_true",
        help="강제 재분석",
    )
    args = parser.parse_args()

    if args.no_skip:
        args.skip = False

    run(args)


if __name__ == "__main__":
    main()
