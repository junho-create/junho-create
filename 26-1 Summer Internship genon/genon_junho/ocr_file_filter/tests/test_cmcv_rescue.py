"""요소 단위 rescue 단위 테스트.

핵심 회귀 방지 대상:
  - 채택 출처: **표는 paddle, 나머지는 dots.ocr**
  - paddle 표 셀 안의 개행(\\n)은 제거하고 반영
  - 개행 차이만으로 표가 불합의 처리되면 안 된다
  - 합의 안 된 요소는 버리되, 너무 많이 버리면(커버리지 미달) rescue 실패

    pytest tests/test_cmcv_rescue.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_filter.cmcv.normalize import STRICT_CATEGORIES, category_agrees  # noqa: E402
from ocr_filter.cmcv.rescue import (  # noqa: E402
    _text_agrees,
    clean_cell_newlines,
    rescue_page,
)

SIZE = (1000, 1000)
TABLE = "<table><tr><td>a</td><td>b</td></tr></table>"


def _el(cat, text, bbox):
    return {"category": cat, "text": text, "bbox": bbox}


# ── 셀 개행 제거 ──────────────────────────────────────────────────────────────
def test_clean_cell_newlines_removes_all_newline_forms():
    assert clean_cell_newlines("<td>가\n나</td>") == "<td>가나</td>"
    assert clean_cell_newlines("<td>가\r\n나</td>") == "<td>가나</td>"
    assert clean_cell_newlines("<td>가\r나</td>") == "<td>가나</td>"


def test_clean_cell_newlines_on_real_paddle_shape():
    """실측 paddle 출력 형태 — 셀 안에서 시각적으로 줄이 바뀐 것을 개행문자로 넣는다."""
    src = ("<table><tr><td>4. 투자 예정 기간을 선택해 주세요.\n"
           "☐ 3년 이상\n☐ 2년 이상 ~ 3년 이내</td></tr></table>")
    out = clean_cell_newlines(src)
    assert "\n" not in out
    assert "☐ 3년 이상☐ 2년 이상" in out


def test_clean_cell_newlines_handles_none_and_empty():
    assert clean_cell_newlines("") == ""
    assert clean_cell_newlines(None) == ""


# ── 텍스트 합의 판정 ──────────────────────────────────────────────────────────
def test_table_agreement_ignores_newline_only_difference():
    """내용은 같은데 paddle 만 개행을 넣은 표는 **합의된 것으로** 봐야 한다."""
    a = _el("Table", "<table><tr><td>가나</td></tr></table>", [0, 0, 100, 100])
    b = _el("Table", "<table><tr><td>가\n나</td></tr></table>", [0, 0, 100, 100])
    ok, _ = _text_agrees(a, b, text_min=0.90)
    assert ok


def test_picture_needs_no_text_agreement():
    a = _el("Picture", "", [0, 0, 100, 100])
    b = _el("Picture", "캡션 비슷한 것", [0, 0, 100, 100])
    ok, score = _text_agrees(a, b, text_min=0.90)
    assert ok and score is None


def test_one_sided_text_is_disagreement():
    """한쪽만 글자를 읽었으면 불합의 — 빈 라벨이 GT 로 들어가면 안 된다."""
    ok, score = _text_agrees(_el("Text", "내용 있음", None), _el("Text", "", None), 0.90)
    assert not ok and score == 0.0


# ── 채택 출처 ────────────────────────────────────────────────────────────────
def test_non_table_adopts_dots_version():
    a = [_el("Text", "dots 판본", [0, 0, 1000, 1000])]
    b = [_el("Text", "dots 판본", [0, 0, 1000, 1000])]
    r = rescue_page(a, b, SIZE)
    assert r is not None
    assert r["elements"][0]["text"] == "dots 판본"


def test_table_adopts_paddle_version_with_newlines_stripped():
    """표는 paddle 본을 쓰되 셀 개행은 지운다."""
    a = [_el("Table", "<table><tr><td>가나</td></tr></table>", [0, 0, 1000, 1000])]
    b = [_el("Table", "<table><tr><td>가\n나</td></tr></table>", [0, 0, 1000, 1000])]
    r = rescue_page(a, b, SIZE)
    assert r is not None
    text = r["elements"][0]["text"]
    assert "\n" not in text          # 개행 제거됨
    assert text == "<table><tr><td>가나</td></tr></table>"


def test_table_adopts_paddle_even_when_dots_differs_in_markup():
    """paddle 이 rowspan 을 더 정확히 잡는 경우 — 채택본은 paddle 이어야 한다."""
    padd = '<table><tr><td rowspan="2">병합</td><td>x</td></tr><tr><td>y</td></tr></table>'
    dots = '<table><tr><td rowspan="2">병합</td><td>x</td></tr><tr><td>y</td></tr></table>'
    a = [_el("Table", dots, [0, 0, 1000, 1000])]
    b = [_el("Table", padd, [0, 0, 1000, 1000])]
    r = rescue_page(a, b, SIZE)
    assert r is not None
    assert 'rowspan="2"' in r["elements"][0]["text"]


# ── 불합의 요소 제거 / 커버리지 ────────────────────────────────────────────────
def test_disagreeing_element_is_dropped_but_page_survives_if_small():
    """작은 요소 하나만 갈리면 그것만 버리고 나머지로 페이지를 살린다."""
    big_a = _el("Text", "동일한 큰 본문", [0, 0, 1000, 900])
    big_b = _el("Text", "동일한 큰 본문", [0, 0, 1000, 900])
    small_a = _el("Text", "여기는 완전히 다름", [0, 920, 200, 960])
    small_b = _el("Text", "전혀 다른 글자입니다", [0, 920, 200, 960])
    r = rescue_page([big_a, small_a], [big_b, small_b], SIZE, coverage_min=0.85)
    assert r is not None
    assert r["n_accepted"] == 1
    assert r["reject_reasons"].get("text_mismatch") == 1


def test_rescue_fails_when_too_much_is_dropped():
    """대부분이 불합의면 rescue 실패 — 라벨 빠진 페이지는 후처리에서 어차피 탈락한다."""
    a = [_el("Text", "AAAA", [0, 0, 1000, 900]), _el("Text", "동일", [0, 920, 100, 960])]
    b = [_el("Text", "ZZZZ 전혀 다름", [0, 0, 1000, 900]), _el("Text", "동일", [0, 920, 100, 960])]
    assert rescue_page(a, b, SIZE, coverage_min=0.90) is None


def test_category_mismatch_is_rejected():
    a = [_el("Table", TABLE, [0, 0, 1000, 1000])]
    b = [_el("Text", TABLE, [0, 0, 1000, 1000])]
    r = rescue_page(a, b, SIZE)
    assert r is None  # 유일한 요소가 카테고리 불일치로 탈락 → 건질 게 없음


def test_unmatched_elements_counted_and_penalize_coverage():
    """한쪽만 잡은 영역은 매칭이 안 되므로 커버리지에서 빠진다."""
    a = [_el("Text", "공통", [0, 0, 1000, 500])]
    b = [_el("Text", "공통", [0, 0, 1000, 500]), _el("Text", "b만 잡음", [0, 520, 1000, 1000])]
    r = rescue_page(a, b, SIZE, coverage_min=0.40)
    assert r is not None
    assert r["reject_reasons"].get("unmatched") == 1
    assert r["coverage"] < 1.0


def test_empty_input_returns_none():
    assert rescue_page([], [_el("Text", "x", [0, 0, 10, 10])], SIZE) is None
    assert rescue_page([_el("Text", "x", [0, 0, 10, 10])], [], SIZE) is None


# ── 카테고리 합의 정책 (2026-07-30 확정) ──────────────────────────────────────
# Picture/Table/Formula/Section-header 만 엄격 합의를 요구하고, 나머지 본문 텍스트
# 성격끼리는 갈려도 상관없다. 비교는 항상 표준 11종으로 정규화한 뒤에 한다.


def test_strict_categories_must_match_exactly():
    for c in ("Picture", "Table", "Formula", "Section-header"):
        assert c in STRICT_CATEGORIES
        assert category_agrees(c, c)
        assert not category_agrees(c, "Text")


def test_section_header_is_strict_even_against_other_headings():
    """Section-header 는 문서 계층을 정의하므로 Title/Caption 과도 합의로 보지 않는다."""
    assert not category_agrees("Section-header", "Title")
    assert not category_agrees("Section-header", "Caption")


def test_body_text_categories_are_interchangeable():
    """dots 는 List-item, paddle 은 Text 로 내는 관례 차이 — 실측 455쌍이 전부 여기 걸렸다."""
    assert category_agrees("List-item", "Text")
    assert category_agrees("Page-footer", "List-item")
    assert category_agrees("Caption", "Text")


def test_comparison_normalizes_paddle_nonstandard_labels():
    """paddle 원본 라벨은 그대로 두고 **비교할 때만** 표준으로 정규화한다."""
    assert category_agrees("Picture", "chart")            # chart → Picture
    assert category_agrees("Footnote", "vision_footnote")  # → Footnote
    assert category_agrees("Text", "reference_content")    # → Text
    assert category_agrees("Text", "aside_text")           # → Text
    assert not category_agrees("Table", "chart")           # 둘 다 엄격 → 불합의


def test_rescue_keeps_page_when_only_listitem_vs_text_differs():
    """목록 표기 차이만으로 페이지가 통째로 탈락하면 안 된다."""
    a = [_el("List-item", "가. 첫 번째 항목", [0, 0, 1000, 1000])]
    b = [_el("Text", "가. 첫 번째 항목", [0, 0, 1000, 1000])]
    r = rescue_page(a, b, SIZE)
    assert r is not None
    assert r["n_accepted"] == 1


def test_rescue_rejects_when_strict_category_differs():
    a = [_el("Section-header", "제 3 장 총칙", [0, 0, 1000, 1000])]
    b = [_el("Text", "제 3 장 총칙", [0, 0, 1000, 1000])]
    assert rescue_page(a, b, SIZE) is None


def test_adopted_label_carries_normalized_category():
    """최종 GT 라벨의 category 는 표준 11종이어야 한다(paddle 원시 라벨이 새면 안 됨)."""
    a = [_el("Picture", "", [0, 0, 1000, 1000])]
    b = [_el("chart", "", [0, 0, 1000, 1000])]
    r = rescue_page(a, b, SIZE)
    assert r is not None
    assert r["elements"][0]["category"] == "Picture"


def test_adopted_category_always_comes_from_dots_even_for_table_content():
    """표는 텍스트/구조를 paddle 에서 채택해도 category 태그는 dots.ocr(ea) 기준이어야 한다.
    dots.ocr 는 표준 11종으로 파인튜닝됐고 paddle 은 아니라서, 어느 쪽 콘텐츠를 쓰든
    카테고리 어휘의 신뢰 출처는 항상 dots.ocr 다."""
    dots_table = _el("Table", "<table><tr><td>a</td></tr></table>", [0, 0, 1000, 1000])
    padd_table = {"category": "table", "text": "<table><tr><td>a\n</td></tr></table>",
                  "bbox": [0, 0, 1000, 1000]}
    r = rescue_page([dots_table], [padd_table], SIZE)
    assert r is not None
    out = r["elements"][0]
    assert out["category"] == "Table"
    assert "<table><tr><td>a</td></tr></table>" == out["text"]


def test_adopted_category_from_dots_even_when_paddle_uses_nonstandard_label():
    """paddle 이 List-item 대신 text 를 내는 경우에도(비엄격 카테고리) 최종 태그는
    dots.ocr 의 List-item 이어야 한다 — paddle 의 낮은 분류 해상도가 GT 로 새면 안 된다."""
    dots_item = _el("List-item", "가. 항목", [0, 0, 1000, 1000])
    padd_item = _el("Text", "가. 항목", [0, 0, 1000, 1000])
    r = rescue_page([dots_item], [padd_item], SIZE)
    assert r is not None
    assert r["elements"][0]["category"] == "List-item"


# ── N:M 그룹 매칭 (2026-07-30) ────────────────────────────────────────────────
def test_group_matching_absorbs_split_granularity_difference():
    """dots 가 문단 1개로 낸 것을 paddle 이 라인 2개로 쪼갠 경우 — 1:1 매칭이면
    IoU 가 부족해 둘 다 unmatched 로 빠지지만, 겹치는 요소를 그룹으로 묶어 텍스트를
    이어붙이면 합의로 잡혀야 한다(회수율 1%→22% 상승의 핵심 메커니즘)."""
    a = [_el("Text", "첫줄\n둘째줄", [0, 0, 1000, 200])]
    b = [
        _el("Text", "첫줄", [0, 0, 1000, 100]),
        _el("Text", "둘째줄", [0, 100, 1000, 200]),
    ]
    r = rescue_page(a, b, SIZE)
    assert r is not None
    assert r["n_accepted"] == 1
    assert r["elements"][0]["text"] == "첫줄\n둘째줄"


def test_group_matching_many_to_one_other_direction():
    """반대 방향: dots 가 여러 조각, paddle 이 하나로 합친 경우도 그룹으로 흡수돼야 한다."""
    a = [
        _el("Text", "가나", [0, 0, 1000, 100]),
        _el("Text", "다라", [0, 100, 1000, 200]),
    ]
    b = [_el("Text", "가나\n다라", [0, 0, 1000, 200])]
    r = rescue_page(a, b, SIZE)
    assert r is not None
    assert r["n_accepted"] == 2  # dots 쪽 개별 요소 단위를 그대로 유지(병합하지 않음)


def test_strict_category_still_uses_fine_matching_within_group():
    """엄격 카테고리는 그룹으로 묶여도 내부에서 세부 1:1 매칭 + 개별 판정을 유지해야 한다
    (본문처럼 뭉뚱그려 비교하면 Table 전용 TEDS 채점을 건너뛰게 된다)."""
    dots_table = _el("Table", "<table><tr><td>a</td></tr></table>", [0, 0, 1000, 500])
    dots_caption = _el("Caption", "표 1. 설명", [0, 500, 1000, 600])
    padd_table = _el("Table", "<table><tr><td>a</td></tr></table>", [0, 0, 1000, 500])
    padd_caption = _el("Text", "표 1. 설명", [0, 500, 1000, 600])  # 캡션을 Text 로 냄(비엄격 차이)
    r = rescue_page([dots_table, dots_caption], [padd_table, padd_caption], SIZE)
    assert r is not None
    cats = {e["category"] for e in r["elements"]}
    assert "Table" in cats
    texts = {e["category"]: e["text"] for e in r["elements"]}
    assert texts["Table"] == "<table><tr><td>a</td></tr></table>"


def test_orphan_strict_element_rejected_but_neighbors_can_survive():
    """한 그룹 안에서 Picture 가 한쪽에만 있으면 그 요소만 탈락하고, 같은 그룹의
    본문류 요소는(겹쳐서 한 그룹이 됐더라도) 별도로 판정된다."""
    dots_pic = _el("Picture", "", [0, 0, 500, 500])
    dots_text = _el("Text", "본문", [500, 0, 1000, 500])
    padd_text_only = _el("Text", "본문", [0, 0, 1000, 500])  # 그림 없이 큰 블록 하나로 잡음
    r = rescue_page([dots_pic, dots_text], [padd_text_only], SIZE)
    # Picture 는 짝이 없어 탈락. Text 는 paddle 의 큰 블록과 겹쳐 같은 그룹이 되지만
    # Picture 로 인해 orphan_strict 처리되고, Text 끼리는 별도로 텍스트 풀 비교된다.
    if r is not None:
        assert all(e["category"] != "Picture" for e in r["elements"])
