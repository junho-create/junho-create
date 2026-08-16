"""정규화 편집거리 기반 텍스트 유사도. 본문/레이아웃 텍스트(Text/Title/... 블록) 비교용.

    edit_distance_score(gt, pred) -> [0, 1], 1 이 완전 일치.
"""

from __future__ import annotations


def levenshtein(a: str, b: str) -> int:
    """표준 편집거리 (삽입/삭제/치환 비용 각 1). O(len(a)*len(b)) 메모리 절약형(2행)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,        # 삭제
                curr[j - 1] + 1,    # 삽입
                prev[j - 1] + cost,  # 치환/일치
            )
        prev = curr
    return prev[-1]


def edit_distance_score(gt: str, pred: str) -> float:
    """1 - normalized_levenshtein. 둘 다 빈 문자열이면 1.0 (완전 일치 취급)."""
    if not gt and not pred:
        return 1.0
    dist = levenshtein(gt, pred)
    denom = max(len(gt), len(pred))
    return 1.0 - dist / denom
