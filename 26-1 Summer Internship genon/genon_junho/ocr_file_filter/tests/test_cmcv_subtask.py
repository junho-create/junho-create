"""subtask 별 일관성 판정 (MinerU2.5-Pro §3.2 의 task-specific metrics 방식) 단위 테스트.

핵심 회귀 방지 대상:
  - 표/수식 매칭이 **등장 순서가 아니라 bbox** 기준이어야 한다 (순서매칭은 개수가 하나만
    달라도 뒤가 전부 밀려 0점 — 실측 Table 슬롯 중앙값 0.000 의 원인)
  - 한 페이지가 subtask 마다 다른 티어를 가질 수 있어야 한다
  - 페이지 대표 티어는 **가장 나쁜 subtask** 를 따라야 한다

    pytest tests/test_cmcv_subtask.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_filter.cmcv.run import (  # noqa: E402
    SUBTASKS,
    _bbox_iou,
    _category_subtask_score,
    _items_with_boxes,
    _match_by_bbox,
    _page_tier,
    _subtask_pair_scores,
    _tier_from_pair_scores,
)
from ocr_filter.metrics import SCORERS  # noqa: E402

AGREE_MIN = 0.85
TABLE_A = "<table><tr><td>a</td><td>b</td></tr></table>"
TABLE_B = "<table><tr><td>a</td><td>b</td></tr></table>"


def _el(cat, text, bbox):
    return {"category": cat, "text": text, "bbox": bbox}


# ── bbox 매칭 ────────────────────────────────────────────────────────────────
def test_bbox_iou_basic():
    assert _bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert _bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert 0.0 < _bbox_iou([0, 0, 10, 10], [5, 5, 15, 15]) < 1.0


def test_bbox_iou_handles_reversed_coords():
    # 모델이 x1<x0 로 뒤집힌 bbox 를 내는 경우가 있다 — 정규화해서 같은 박스로 봐야 한다.
    assert _bbox_iou([10, 10, 0, 0], [0, 0, 10, 10]) == 1.0


def test_match_by_bbox_pairs_overlapping_not_by_order():
    """A 는 표 1개, B 는 표 2개인데 B 의 **두 번째**가 A 와 같은 자리인 경우.
    순서매칭이면 A[0] 이 B[0] 과 짝지어져 오답이 되지만, bbox 매칭은 B[1] 과 짝지어야 한다."""
    a = [(_el("Table", "correct", [0, 500, 100, 600]), [0, 0.5, 0.1, 0.6])]
    b = [
        (_el("Table", "other", [0, 0, 100, 100]), [0, 0, 0.1, 0.1]),
        (_el("Table", "correct", [0, 500, 100, 600]), [0, 0.5, 0.1, 0.6]),
    ]
    pairs, a_miss, b_miss = _match_by_bbox(a, b)
    assert len(pairs) == 1
    assert pairs[0][1]["text"] == "correct"  # 순서가 아니라 위치로 짝지었다
    assert (a_miss, b_miss) == (0, 1)


def test_match_by_bbox_greedy_prefers_highest_iou():
    a = [(_el("Table", "x", None), [0, 0, 1.0, 1.0])]
    b = [
        (_el("Table", "loose", None), [0, 0, 0.5, 0.5]),
        (_el("Table", "tight", None), [0, 0, 0.95, 0.95]),
    ]
    pairs, _, _ = _match_by_bbox(a, b)
    assert pairs[0][1]["text"] == "tight"


# ── 카테고리 subtask 점수 ─────────────────────────────────────────────────────
def test_category_score_none_when_neither_side_has_category():
    a = [(_el("Text", "hi", None), [0, 0, 1, 1])]
    b = [(_el("Text", "hi", None), [0, 0, 1, 1])]
    assert _category_subtask_score(a, b, "Table", SCORERS["teds"]) is None


def test_category_score_perfect_when_same_table_same_place():
    a = [(_el("Table", TABLE_A, None), [0, 0, 1, 1])]
    b = [(_el("Table", TABLE_B, None), [0, 0, 1, 1])]
    assert _category_subtask_score(a, b, "Table", SCORERS["teds"]) == 1.0


def test_category_score_penalizes_unmatched_table():
    """한쪽에만 있는 표(누락/환각)는 0점으로 집계돼 평균을 끌어내려야 한다."""
    a = [(_el("Table", TABLE_A, None), [0, 0, 0.5, 0.5])]
    b = [
        (_el("Table", TABLE_B, None), [0, 0, 0.5, 0.5]),
        (_el("Table", TABLE_B, None), [0.6, 0.6, 1.0, 1.0]),  # b 에만 있는 표
    ]
    score = _category_subtask_score(a, b, "Table", SCORERS["teds"])
    assert score == 0.5  # (1.0 매칭 + 0.0 미매칭) / 2


def test_category_score_survives_count_mismatch_that_order_matching_would_zero():
    """개수가 다르고 순서가 어긋나도, 겹치는 표끼리는 제대로 채점돼야 한다
    (등장 순서 매칭이었다면 전부 어긋나 0에 가까웠을 배치)."""
    a = [
        (_el("Table", TABLE_A, None), [0, 0.6, 1.0, 1.0]),   # 아래쪽 표만 잡음
    ]
    b = [
        (_el("Table", "<table><tr><td>zzz</td></tr></table>", None), [0, 0, 1.0, 0.4]),
        (_el("Table", TABLE_B, None), [0, 0.6, 1.0, 1.0]),
    ]
    score = _category_subtask_score(a, b, "Table", SCORERS["teds"])
    assert score == 0.5  # 겹치는 짝은 1.0, b 의 미매칭 1개가 0.0


# ── subtask 점수 묶음 ─────────────────────────────────────────────────────────
def test_subtask_scores_exclude_table_text_from_text_subtask():
    """text subtask 는 Table/Formula 를 빼고 본문만 비교해야 한다 — 표 HTML 이 섞이면
    표 전용 지표(TEDS)와 중복 계산되고, 표가 큰 페이지에서 본문 신호가 묻힌다."""
    a_els = [_el("Text", "본문 동일", [0, 0, 100, 50]),
             _el("Table", TABLE_A, [0, 60, 100, 200])]
    b_els = [_el("Text", "본문 동일", [0, 0, 100, 50]),
             _el("Table", "<table><tr><td>완전히</td><td>다름</td></tr></table>", [0, 60, 100, 200])]
    a_items = _items_with_boxes(a_els, "pixel", 100, 200)
    b_items = _items_with_boxes(b_els, "pixel", 100, 200)
    boxes_a = [b for _, b in a_items]
    boxes_b = [b for _, b in b_items]
    s = _subtask_pair_scores(a_els, b_els, a_items, b_items, boxes_a, boxes_b)
    assert s["text"] == 1.0          # 본문은 완전 일치
    assert s["table"] < 1.0          # 표는 불일치 — 서로 독립적으로 잡힌다
    assert s["formula"] is None      # 수식 없음


def test_subtask_scores_layout_always_present():
    a_els = [_el("Text", "x", [0, 0, 100, 100])]
    b_els = [_el("Text", "y", [0, 0, 100, 100])]
    a_items = _items_with_boxes(a_els, "pixel", 100, 100)
    b_items = _items_with_boxes(b_els, "pixel", 100, 100)
    s = _subtask_pair_scores(a_els, b_els, a_items, b_items,
                             [b for _, b in a_items], [b for _, b in b_items])
    assert s["layout"] == 1.0
    assert set(s) == set(SUBTASKS)


# ── 티어 판정 ────────────────────────────────────────────────────────────────
def test_tier_easy_when_target_agrees_with_one_external():
    assert _tier_from_pair_scores(0.95, 0.10, 0.10, AGREE_MIN) == "Easy"
    assert _tier_from_pair_scores(0.10, 0.95, 0.10, AGREE_MIN) == "Easy"


def test_tier_medium_when_only_externals_agree():
    assert _tier_from_pair_scores(0.10, 0.10, 0.95, AGREE_MIN) == "Medium"


def test_tier_hard_when_all_disagree():
    assert _tier_from_pair_scores(0.10, 0.10, 0.10, AGREE_MIN) == "Hard"


def test_tier_none_when_subtask_absent():
    """해당 subtask 요소가 아무 모델에도 없으면 판정 대상이 아니다(Hard 로 몰면 안 됨)."""
    assert _tier_from_pair_scores(None, None, None, AGREE_MIN) is None


def test_tier_ignores_missing_pairs():
    # 한 쌍만 계산 가능한 경우에도 그 쌍으로 판정할 수 있어야 한다.
    assert _tier_from_pair_scores(None, None, 0.95, AGREE_MIN) == "Medium"
    assert _tier_from_pair_scores(0.95, None, None, AGREE_MIN) == "Easy"


# ── 페이지 대표 티어 ──────────────────────────────────────────────────────────
def test_page_tier_takes_worst_subtask():
    """표 하나가 Hard 인 페이지를 Easy 라고 부르면 그 표가 무검증 GT 가 된다."""
    assert _page_tier({"layout": "Easy", "text": "Easy", "table": "Hard", "formula": None}) == "Hard"
    assert _page_tier({"layout": "Easy", "text": "Medium", "table": None,
                       "formula": None}) == "Medium"
    assert _page_tier({"layout": "Easy", "text": "Easy", "table": "Easy",
                       "formula": None}) == "Easy"


def test_page_tier_defaults_hard_when_nothing_judged():
    assert _page_tier({s: None for s in SUBTASKS}) == "Hard"
