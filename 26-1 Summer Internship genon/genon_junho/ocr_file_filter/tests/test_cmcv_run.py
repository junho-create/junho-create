"""cmcv/run.py 단위 테스트: 티어는 쌍별(pairwise) 일치/불일치 조합 규칙으로 정해져야 함
(평균 아님), gt_score/teds_score 는 참고용이고 gt 없는 레코드에서는 None.

    pytest tests/test_cmcv_run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_filter.cmcv.run import (  # noqa: E402
    _blended_pair_score,
    _elements_to_html,
    _matched_category_score,
    _pairwise_scores,
    _score_against_gt,
    _tier_and_pseudo_label,
)
from ocr_filter.cmcv.normalize import full_text  # noqa: E402
from ocr_filter.io.schema import Record  # noqa: E402
from ocr_filter.metrics import SCORERS  # noqa: E402

AGREE_MIN = 0.85


def _els(text: str) -> list[dict]:
    return [{"category": "Text", "text": text, "bbox": None}]


def test_score_against_gt_returns_none_when_gt_missing():
    record = Record(id="x", image_path="x.png", gt=None, source_type="layout")
    assert _score_against_gt(record, _els("a")) == (None, None)


def test_score_against_gt_returns_none_when_gt_empty_list():
    # 미래의 GT 없는 신규 데이터를 흉내: io 단계가 gt=[] 로 채워 넣는 경우도 "GT 없음" 취급.
    record = Record(id="x", image_path="x.png", gt=[], source_type="layout")
    assert _score_against_gt(record, _els("a")) == (None, None)


def test_score_against_gt_computes_when_gt_present():
    record = Record(id="x", image_path="x.png", gt=_els("hello"), source_type="layout")
    gt_score, teds_bonus = _score_against_gt(record, _els("hello"))
    assert gt_score == 1.0
    assert teds_bonus is None  # layout 레코드는 teds 보너스 없음(표 레코드 전용)


def test_pairwise_scores_computes_all_three_pairs():
    elements = {"target": _els("hello world"), "external_a": _els("hello world"),
                "external_b": _els("totally different")}
    pw = _pairwise_scores(elements, "x.png")
    assert set(pw) == {"target_dots", "target_paddle", "dots_paddle"}
    assert pw["target_dots"] == 1.0
    assert pw["target_paddle"] < 1.0


def test_easy_when_target_agrees_with_at_least_one_external():
    # target 은 dots.ocr 이랑만 맞고 paddle 이랑은 안 맞음 → 그래도 Easy
    # (평균 내면 (1.0 + 낮음 + 낮음)/3 이 0.85 밑으로 떨어져 Hard 로 오분류될 수 있는 케이스).
    elements = {"target": _els("hello world"), "external_a": _els("hello world"),
                "external_b": _els("something totally unrelated here")}
    pw = _pairwise_scores(elements, "x.png")
    tier, pseudo, _ = _tier_and_pseudo_label(pw, elements, AGREE_MIN)
    assert tier == "Easy"
    assert pseudo is None


def test_medium_when_externals_agree_but_target_diverges():
    # dots.ocr-paddle 은 서로 맞는데 target 만 튐 → Medium, dots.ocr 출력이 pseudo-label.
    elements = {"target": _els("완전히 다른 잘못된 결과"), "external_a": _els("정답 텍스트"),
                "external_b": _els("정답 텍스트")}
    pw = _pairwise_scores(elements, "x.png")
    tier, pseudo, _ = _tier_and_pseudo_label(pw, elements, AGREE_MIN)
    assert tier == "Medium"
    assert pseudo == elements["external_a"]


def test_medium_text_gate_demotes_layout_only_agreement_to_hard():
    # 블렌드(dots_paddle)는 레이아웃 슬롯(page_iou/generic_teds) 덕에 agree_min 을 넘지만
    # 정작 텍스트끼리는 크게 다른 케이스 — pseudo-label 텍스트가 미검증으로 GT 가 되는
    # 것을 텍스트 전용 게이트(text_min)가 차단하고 Hard(rescue/judge 경로)로 보내야 한다.
    elements = {"target": _els("완전히 다른 결과"), "external_a": _els("정답 텍스트라고 주장"),
                "external_b": _els("전혀 다른 내용을 읽음")}
    pw = {"target_dots": 0.3, "target_paddle": 0.3, "dots_paddle": 0.86}  # 블렌드는 통과 가정
    tier, pseudo, text_score = _tier_and_pseudo_label(pw, elements, AGREE_MIN)
    assert tier == "Hard"
    assert pseudo is None
    assert text_score is not None and text_score < 0.90  # 게이트 판정값이 기록으로 남는다


def test_medium_text_gate_passes_when_texts_agree():
    elements = {"target": _els("완전히 다른 잘못된 결과"), "external_a": _els("정답 텍스트"),
                "external_b": _els("정답 텍스트")}
    pw = {"target_dots": 0.3, "target_paddle": 0.3, "dots_paddle": 0.95}
    tier, pseudo, text_score = _tier_and_pseudo_label(pw, elements, AGREE_MIN)
    assert tier == "Medium"
    assert pseudo == elements["external_a"]
    assert text_score == 1.0


def test_hard_when_all_three_disagree():
    elements = {"target": _els("aaa"), "external_a": _els("bbb"), "external_b": _els("ccc")}
    pw = _pairwise_scores(elements, "x.png")
    tier, pseudo, _ = _tier_and_pseudo_label(pw, elements, AGREE_MIN)
    assert tier == "Hard"
    assert pseudo is None


def test_easy_takes_priority_when_everything_agrees():
    elements = {"target": _els("same"), "external_a": _els("same"), "external_b": _els("same")}
    pw = _pairwise_scores(elements, "x.png")
    tier, pseudo, _ = _tier_and_pseudo_label(pw, elements, AGREE_MIN)
    assert tier == "Easy"
    assert pseudo is None


def test_elements_to_html_wraps_by_category():
    # 카테고리가 태그 이름 자체가 되어야 rename_cost(태그 비교)가 카테고리 불일치를 반영한다.
    elements = [{"category": "Title", "text": "hi", "bbox": None}]
    html = _elements_to_html(elements)
    assert "<title>hi</title>" in html


def test_elements_to_html_keeps_table_html_raw():
    elements = [{"category": "Table", "text": "<table><tr><td>a</td></tr></table>", "bbox": None}]
    html = _elements_to_html(elements)
    assert "<table><tr><td>a</td></tr></table>" in html
    assert "<p><table" not in html  # Table 은 <p> 로 안 감쌈


def test_target_dots_blends_text_and_structure_same_category():
    # 카테고리(태그)는 완전히 같은데 텍스트만 살짝 다른 경우: 순수 edit_distance 보다
    # target_dots 최종 점수가 (edit_distance + generic_teds + page_iou)/3 로 완화되는지 확인.
    # bbox=None(테스트용 가짜 엘리먼트)이라 page_iou 는 둘 다 박스 없음 → 1.0.
    same_category = {"target": _els("hello wrold"), "external_a": _els("hello world"),
                      "external_b": _els("totally different")}
    pw = _pairwise_scores(same_category, "x.png")
    text_only = SCORERS["edit_distance"]("hello wrold", "hello world")
    assert pw["target_dots"] > text_only
    assert pw["target_dots"] == (
        text_only
        + SCORERS["generic_teds"](
            _elements_to_html(same_category["target"]),
            _elements_to_html(same_category["external_a"]),
        )
        + 1.0
    ) / 3


def test_borderline_near_miss_can_cross_into_easy_with_teds():
    # 실측 사례(2026-07-13)를 흉내: 대부분의 엘리먼트(레이아웃/카테고리)는 target 과
    # dots.ocr 이 완전히 일치하고, 딱 하나의 Text 블록만 완전히 다른 문장으로 갈렸다.
    # 페이지 전체를 이어붙인 순수 텍스트 edit_distance 로는 그 한 블록의 글자수 비중 때문에
    # agree_min 바로 밑(0.847)으로 떨어지지만, generic_teds 는 나머지 다 맞는 엘리먼트들을
    # 0-비용으로 매칭시켜 노드 단위로는 그 한 블록의 영향이 훨씬 작아져 target_dots 블렌드가
    # agree_min 을 넘는다 — paddle 은 여전히 안 맞으니 target_paddle 로는 Easy 가 안 되고
    # target_dots 블렌드 덕분에만 Easy 로 판정돼야 한다.
    common = [{"category": "Title", "text": "Quarterly Report", "bbox": None}]
    common += [{"category": "Text", "text": f"Section {i} ok", "bbox": None} for i in range(25)]
    target = common + _els(
        "Revenue increased by twelve percent this quarter compared to prior year figures overall",
    )
    dots = common + _els(
        "Zephyr owl mountain velvet quantum banana orchestra lighthouse gravel telescope",
    )
    paddle = _els("totally unrelated paddle output text here")
    elements = {"target": target, "external_a": dots, "external_b": paddle}

    text_only = SCORERS["edit_distance"](full_text(target), full_text(dots))
    assert text_only < AGREE_MIN  # 블렌드 전이면 Medium/Hard 로 떨어졌을 경계 케이스

    pw = _pairwise_scores(elements, "x.png")
    assert pw["target_dots"] >= AGREE_MIN
    assert pw["target_paddle"] < AGREE_MIN  # paddle 로는 여전히 Easy 조건 미충족
    tier, _, _ = _tier_and_pseudo_label(pw, elements, AGREE_MIN)
    assert tier == "Easy"


# ===== 카테고리 라우팅 (Table→TEDS, Formula→CDM, 2026-07-19 추가) =====


def test_matched_category_score_none_when_neither_side_has_category():
    a = [{"category": "Text", "text": "hi", "bbox": None}]
    b = [{"category": "Text", "text": "hi", "bbox": None}]
    assert _matched_category_score(a, b, "Table", lambda x, y: 1.0) is None


def test_matched_category_score_averages_matched_pairs_by_order():
    a = [{"category": "Table", "text": "a1"}, {"category": "Table", "text": "a2"}]
    b = [{"category": "Table", "text": "b1"}, {"category": "Table", "text": "b2"}]
    calls = []

    def scorer(x, y):
        calls.append((x, y))
        return 1.0 if x == "a1" else 0.5

    assert _matched_category_score(a, b, "Table", scorer) == (1.0 + 0.5) / 2
    assert calls == [("a1", "b1"), ("a2", "b2")]  # 순서(인덱스) 기준 매칭


def test_matched_category_score_zero_for_unmatched_pair():
    # 한쪽에만 있는(짝 없는) 요소는 통째로 누락/환각한 것 -- 벌점으로 0점.
    a = [{"category": "Formula", "text": "x^2"}, {"category": "Formula", "text": "y^2"}]
    b = [{"category": "Formula", "text": "x^2"}]
    assert _matched_category_score(a, b, "Formula", lambda x, y: 1.0) == (1.0 + 0.0) / 2


def test_matched_category_score_skips_scorer_none_results():
    # scorer 가 None(예: CDM 렌더 실패, 가짜 점수 방지 정책)을 내면 그 쌍은 집계에서 제외.
    a = [{"category": "Formula", "text": "x^2"}, {"category": "Formula", "text": "y^2"}]
    b = [{"category": "Formula", "text": "x^2"}, {"category": "Formula", "text": "y^2"}]
    results = iter([None, 0.8])
    assert _matched_category_score(a, b, "Formula", lambda x, y: next(results)) == 0.8


def test_matched_category_score_all_none_returns_none():
    a = [{"category": "Formula", "text": "x^2"}]
    b = [{"category": "Formula", "text": "x^2"}]
    assert _matched_category_score(a, b, "Formula", lambda x, y: None) is None


def test_blended_pair_score_unaffected_without_table_or_formula():
    # 표/수식이 없는 페이지는 기존 3-way 블렌드(text+generic_teds+page_iou)/3 그대로 나와야
    # 한다 -- 기존 agree_min 캘리브레이션(2026-07-13/14 실측)이 이 리팩터로 안 깨졌는지 확인.
    a = [{"category": "Text", "text": "hello world", "bbox": None}]
    b = [{"category": "Text", "text": "hello wrold", "bbox": None}]
    got = _blended_pair_score(a, b, [], [])
    expected = (
        SCORERS["edit_distance"]("hello world", "hello wrold")
        + SCORERS["generic_teds"](_elements_to_html(a), _elements_to_html(b))
        + SCORERS["page_iou"]([], [])
    ) / 3
    assert got == expected


def test_blended_pair_score_adds_table_slot_when_either_side_has_table(monkeypatch):
    monkeypatch.setitem(SCORERS, "teds", lambda gt, pred: 1.0)
    a = [{"category": "Table", "text": "<table><tr><td>a</td></tr></table>", "bbox": None}]
    b = [{"category": "Table", "text": "<table><tr><td>a</td></tr></table>", "bbox": None}]
    got = _blended_pair_score(a, b, [], [])
    text_score = SCORERS["edit_distance"](full_text(a), full_text(b))
    teds_score = SCORERS["generic_teds"](_elements_to_html(a), _elements_to_html(b))
    iou_score = SCORERS["page_iou"]([], [])
    expected = (text_score + teds_score + iou_score + 1.0) / 4  # table_score 슬롯이 추가돼 4-way
    assert got == expected


def test_blended_pair_score_adds_formula_slot_when_either_side_has_formula(monkeypatch):
    monkeypatch.setitem(SCORERS, "cdm", lambda gt, pred: 0.9)
    a = [{"category": "Formula", "text": "x^2", "bbox": None}]
    b = [{"category": "Formula", "text": "x^2", "bbox": None}]
    got = _blended_pair_score(a, b, [], [])
    text_score = SCORERS["edit_distance"](full_text(a), full_text(b))
    teds_score = SCORERS["generic_teds"](_elements_to_html(a), _elements_to_html(b))
    iou_score = SCORERS["page_iou"]([], [])
    expected = (text_score + teds_score + iou_score + 0.9) / 4
    assert got == expected


def test_blended_pair_score_formula_slot_skipped_when_cdm_unavailable(monkeypatch):
    # TeX Live 미설치/렌더 실패 등으로 cdm_score 가 항상 None(가짜 점수 방지 정책) -- formula
    # 슬롯 자체가 안 생기고 조용히 기존 3-way 블렌드로 폴백해야 한다(예외 나면 안 됨).
    monkeypatch.setitem(SCORERS, "cdm", lambda gt, pred: None)
    a = [{"category": "Formula", "text": "x^2", "bbox": None}]
    b = [{"category": "Formula", "text": "x^2", "bbox": None}]
    got = _blended_pair_score(a, b, [], [])
    text_score = SCORERS["edit_distance"](full_text(a), full_text(b))
    teds_score = SCORERS["generic_teds"](_elements_to_html(a), _elements_to_html(b))
    iou_score = SCORERS["page_iou"]([], [])
    expected = (text_score + teds_score + iou_score) / 3
    assert got == expected
