"""GT HTML 의 빈 cell 을 marker 로 채우는 유틸 (라이브러리 + CLI).

목적:
    `<td></td>`, `<th></th>` (whitespace only 도 포함) 을 marker token
    (예: `__EMPTY__`) 으로 채워서 모델이 모든 cell position 에 무언가 출력하도록
    학습하게 한다. Inference 후 strip_filler_cells() 로 marker 제거하여
    원본 형식 복원 가능.

학습 흐름:
    1. Train data 의 GT HTML 의 빈 cell → `__EMPTY__` 로 채움
    2. Hint-Completion 등이 채워진 GT 사용
    3. 모델은 빈 cell 도 `<td>__EMPTY__</td>` 로 출력 학습

평가 흐름:
    1. Inference: 모델은 `<td>__EMPTY__</td>` 형태로 출력
    2. strip_filler_cells: marker 제거 → `<td></td>` 복원
    3. 원본 (unmodified) GT 와 TEDS-S 비교

API (라이브러리):
    fill_empty_cells(html, marker)    — 단일 HTML 의 빈 cell 채움
    strip_filler_cells(html, marker)  — 단일 HTML 에서 marker 제거
    count_filled_cells(html, marker)  — marker cell 개수 카운트

CLI (jsonl batch):
    python -m utils.fill_empty_cells \\
        --input data/train.jsonl \\
        --output data/train_filled.jsonl
"""
from __future__ import annotations

import re

# `__EMPTY__` 는 V8.x train data 에서 한 번도 사용된 적 없음 (collision 0)
DEFAULT_MARKER = "__EMPTY__"


# 빈 cell 패턴 — opening tag + (whitespace only) + closing tag
# 예: '<td></td>', '<td  ></td  >', '<td class="x"></td>', '<td>\n  </td>'
_EMPTY_CELL_PATTERN = re.compile(
    r"(<(td|th)\b[^>]*>)(\s*)(</\2\s*>)",
    flags=re.IGNORECASE | re.DOTALL,
)


def fill_empty_cells(html: str, marker: str = DEFAULT_MARKER) -> str:
    """GT HTML 의 빈 <td>/<th> 를 marker 로 채움.

    Args:
        html: 원본 HTML 문자열
        marker: 채울 token (default `__EMPTY__`)

    Returns:
        빈 cell 이 채워진 HTML.

    Examples:
        '<td></td>' → '<td>__EMPTY__</td>'
        '<td>real</td>' → '<td>real</td>'  (변경 X)
        '<td  ></td  >' → '<td  >__EMPTY__</td  >'
        '<td class="x">  </td>' → '<td class="x">__EMPTY__</td>'
    """
    if not html:
        return html

    def _fill(m: re.Match) -> str:
        return m.group(1) + marker + m.group(4)

    return _EMPTY_CELL_PATTERN.sub(_fill, html)


def strip_filler_cells(html: str, marker: str = DEFAULT_MARKER) -> str:
    """Pred HTML 에서 marker only cell 의 content 제거 (cell 자체는 유지).

    Args:
        html: 모델 생성 HTML (marker 포함 가능)
        marker: 제거할 token

    Returns:
        marker 가 제거된 HTML — `<td>__EMPTY__</td>` → `<td></td>`

    Examples:
        '<td>__EMPTY__</td>' → '<td></td>'
        '<td>real</td>' → '<td>real</td>'  (변경 X)
        '<td>real __EMPTY__</td>' → '<td>real __EMPTY__</td>'  (only-marker 가 아니므로 유지)
    """
    if not html:
        return html

    escaped = re.escape(marker)
    pattern = re.compile(
        rf"(<(td|th)\b[^>]*>)\s*{escaped}\s*(</\2\s*>)",
        flags=re.IGNORECASE | re.DOTALL,
    )

    def _strip(m: re.Match) -> str:
        return m.group(1) + m.group(3)

    return pattern.sub(_strip, html)


def count_filled_cells(html: str, marker: str = DEFAULT_MARKER) -> int:
    """marker 로 채워진 cell 개수 (디버깅/검증용)."""
    if not html:
        return 0
    escaped = re.escape(marker)
    pattern = re.compile(
        rf"<(td|th)\b[^>]*>\s*{escaped}\s*</\1\s*>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    return len(pattern.findall(html))


# =============================================================================
# CLI — jsonl batch 변환
# =============================================================================


def _main():
    """CLI: jsonl 의 모든 record gt_html 에 대해 fill_empty_cells 적용.

    각 record 의 gt_html 만 변경. 다른 필드 (image_path, ocr_info, bucket 등) 그대로.
    Bucket 별 fill 분포 통계 출력.
    """
    import argparse
    import json
    from collections import Counter
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="원본 train jsonl")
    ap.add_argument("--output", type=Path, required=True, help="출력 jsonl")
    ap.add_argument("--marker", default=DEFAULT_MARKER,
                    help=f"채울 token (기본 {DEFAULT_MARKER})")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_filled_total = 0
    bucket_filled: Counter = Counter()
    bucket_total: Counter = Counter()

    with args.input.open() as fin, args.output.open("w") as fout:
        for line in fin:
            r = json.loads(line)
            if r.get("_meta"):
                fout.write(line)
                continue

            n_total += 1
            bucket = r.get("bucket", "unknown")
            bucket_total[bucket] += 1

            old_gt = r.get("gt_html", "")
            new_gt = fill_empty_cells(old_gt, args.marker)
            n_filled = count_filled_cells(new_gt, args.marker)
            n_filled_total += n_filled
            if n_filled > 0:
                bucket_filled[bucket] += 1

            r["gt_html"] = new_gt
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[Summary]")
    print(f"  Total records:        {n_total}")
    print(f"  Total cells filled:   {n_filled_total}")
    print(f"  Avg cells/record:     {n_filled_total / max(1, n_total):.1f}")
    print()
    print(f"[Bucket 별 fill 분포 (record 수)]")
    print(f"  {'bucket':25s} {'total':>6s} {'with_fill':>10s} {'%':>6s}")
    for b in sorted(bucket_total):
        t = bucket_total[b]
        f = bucket_filled[b]
        pct = 100 * f / t if t > 0 else 0
        print(f"  {b:25s} {t:6d} {f:10d} {pct:5.1f}%")
    print()
    print(f"[Saved] {args.output}")


if __name__ == "__main__":
    _main()
