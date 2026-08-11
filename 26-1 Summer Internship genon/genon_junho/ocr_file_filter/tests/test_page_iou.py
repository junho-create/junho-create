"""metrics/page_iou.py 단위 테스트.

    pytest tests/test_page_iou.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_filter.metrics.page_iou import page_iou_score  # noqa: E402


def test_both_empty_is_perfect_match():
    assert page_iou_score([], []) == 1.0


def test_one_empty_one_not_is_zero():
    assert page_iou_score([[0.1, 0.1, 0.5, 0.5]], []) == 0.0


def test_identical_boxes_is_one():
    boxes = [[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]]
    assert page_iou_score(boxes, boxes) == 1.0


def test_disjoint_boxes_is_zero():
    a = [[0.0, 0.0, 0.2, 0.2]]
    b = [[0.8, 0.8, 1.0, 1.0]]
    assert page_iou_score(a, b) == 0.0


def test_coarse_vs_fine_split_scores_higher_than_naive_box_iou_would():
    # 논문 Figure 4 취지: 문단 전체를 뭉뚱그린 박스(a) vs 줄 단위로 쪼갠 박스(b, 합치면
    # 거의 같은 영역) — 개별 박스 IoU 매칭이면 낮게 나오지만 PageIoU 는 커버리지 자체가
    # 비슷하므로 높은 점수가 나와야 함.
    a = [[0.1, 0.1, 0.9, 0.5]]
    b = [[0.1, 0.1, 0.9, 0.2], [0.1, 0.2, 0.9, 0.3], [0.1, 0.3, 0.9, 0.4], [0.1, 0.4, 0.9, 0.5]]
    score = page_iou_score(a, b)
    assert score > 0.9


def test_ignores_malformed_boxes():
    assert page_iou_score([None, [0.1, 0.1, 0.5, 0.5]], [[0.1, 0.1, 0.5, 0.5]]) == 1.0


def test_partial_overlap_between_zero_and_one():
    a = [[0.0, 0.0, 0.6, 0.6]]
    b = [[0.4, 0.4, 1.0, 1.0]]
    score = page_iou_score(a, b)
    assert 0.0 < score < 1.0
