"""임의 HTML(표 하나가 아니라 페이지 전체 div 트리 등)에 대한 TEDS.

`teds.py`(표 전용, label 완전일치만 봄)와 달리 이건 **태그는 같은데 텍스트만 살짝 다른
경우를 부분점수로** 처리한다 (tsr_eval/vlm/eval/metrics.py 의 `GenericHTMLTEDS`가 apted +
bs4 + editdistance 로 하는 것과 동일한 발상 — 여기서는 새 의존성 없이 이미 있는
`html_tree.py`/`tree_edit_distance.py`(rename_cost 콜백)로 그대로 재구현).

CMCV 쌍별(pairwise) 동의도 판정에서 순수 텍스트 edit_distance 하나만 쓰면, 레이아웃/구조는
사실상 같은데 글자 몇 개 차이로 문턱값(agree_min) 바로 밑으로 떨어지는 경계 케이스가 잦다
(2026-07-13 실측: target_paddle=0.835처럼 agree_min=0.85에 근소하게 못 미치는 사례 다수
발견). generic TEDS 는 노드 단위로 부분점수를 주기 때문에 그런 경계 케이스에서 좀 더
관대하게 나온다 — `cmcv/run.py` 의 `_pairwise_scores` 가 텍스트 점수와 평균내서 씀.
"""

from __future__ import annotations

from ocr_filter.metrics.edit_distance import edit_distance_score
from ocr_filter.metrics.html_tree import Node, parse_html
from ocr_filter.metrics.tree_edit_distance import tree_edit_distance


def _children(node: Node) -> list:
    return node.children


def _rename_cost(a: Node, b: Node, structure_only: bool) -> float:
    if a.tag != b.tag:
        return 1.0
    if structure_only:
        return 0.0
    if not a.text and not b.text:
        return 0.0
    return 1.0 - edit_distance_score(a.text, b.text)


def generic_teds_score(a_html: str, b_html: str, structure_only: bool = False) -> float:
    """임의 HTML 조각 두 개의 TEDS. 둘 다 비거나 파싱 실패하면 1.0(완전일치 취급,
    `teds.py`/`edit_distance_score`와 같은 관례)."""
    a_tree = parse_html(a_html)
    b_tree = parse_html(b_html)
    if a_tree.size() <= 1 and b_tree.size() <= 1:
        return 1.0

    ted = tree_edit_distance(
        a_tree, b_tree, _children,
        rename_cost=lambda x, y: _rename_cost(x, y, structure_only),
    )
    denom = max(a_tree.size(), b_tree.size())
    return 1.0 - ted / denom
