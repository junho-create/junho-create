"""TEDS / TEDS-S: 표 HTML 트리 편집거리 기반 유사도 (PubTabNet 논문 정의).

    teds_score(gt_html, pred_html)                    # 셀 텍스트까지 비교
    teds_score(gt_html, pred_html, structure_only=True)  # TEDS-S: 구조(태그+span)만

TEDS = 1 - TED(T_gt, T_pred) / max(|T_gt|, |T_pred|)

⚠ 이 구현은 순수 파이썬 Zhang-Shasha(정확하지만 APTED보다 느림, 표 하나 기준이라 충분)로
직접 짠 것 — 원 논문/dp-bench 쪽 TEDS 구현(apted 라이브러리 기반)과 소수점 단위로 다를 수
있다. cmcv 스테이지가 이 함수 시그니처(`(gt, pred) -> float in [0,1]`)에만 의존하도록
짜여 있으면, 나중에 다른 TEDS 구현으로 교체해도 나머지는 안 건드려도 된다.
"""

from __future__ import annotations

from ocr_filter.metrics.html_tree import Node, parse_html
from ocr_filter.metrics.tree_edit_distance import tree_edit_distance


def _children(node: Node) -> list:
    return node.children


def teds_score(gt_html: str, pred_html: str, structure_only: bool = False) -> float:
    gt_tree = parse_html(gt_html)
    pred_tree = parse_html(pred_html)

    if gt_tree.size() <= 1 and pred_tree.size() <= 1:
        return 1.0  # 둘 다 빈/파싱안됨 → 완전 일치 취급 (edit_distance_score 와 동일 관례)

    label_key = "struct_label" if structure_only else "label"

    def label_eq(a: Node, b: Node) -> bool:
        return getattr(a, label_key) == getattr(b, label_key)

    ted = tree_edit_distance(gt_tree, pred_tree, _children, label_eq)
    denom = max(gt_tree.size(), pred_tree.size())
    return 1.0 - ted / denom
