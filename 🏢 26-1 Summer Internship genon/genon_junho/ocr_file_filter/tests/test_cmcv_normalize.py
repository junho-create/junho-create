"""cmcv normalize 단위 테스트: Chandra div-HTML / dots.ocr JSON 파서.

    pytest tests/test_cmcv_normalize.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_filter.cmcv.normalize import (  # noqa: E402
    full_text,
    normalize,
    normalize_category,
    parse_chandra_output,
    parse_dots_ocr_output,
    parse_paddle_output,
    plain_text,
    table_htmls,
)

CHANDRA_SAMPLE = """```html
<div data-bbox="10 10 500 60" data-label="Title">
<p>문서 제목</p>
</div>
<div data-bbox="10 70 500 400" data-label="Table">
<table><tr><td>a</td><td>b</td></tr></table>
</div>
<div data-bbox="10 410 500 450" data-label="Page-footer">
<p>1</p>
</div>
```"""

DOTS_SAMPLE = """여기 결과입니다:
```json
[
  {"bbox": [10, 10, 500, 60], "category": "Title", "text": "문서 제목"},
  {"bbox": [10, 70, 500, 400], "category": "Table", "text": "<table><tr><td>a</td><td>b</td></tr></table>"}
]
```
"""


def test_parse_chandra_output_ignores_draft_inside_think_block():
    # target(Qwen3.5) 은 thinking 모델이라 <think> 안에서 최종 답을 미리 "초안"으로
    # 한 번 써보고 </think> 뒤에 또 쓰는 경우가 흔함 — 이걸 안 걷어내면 요소가 2배로 세짐
    # (2026-07-13 실데이터로 재현: 5개 정답인데 파싱되면 10개로 나왔던 버그).
    raw = (
        "<think>\n"
        "이렇게 써야지: "
        '<div data-bbox="1 1 2 2" data-label="Title">\n<p>초안</p>\n</div>\n'
        "</think>\n"
        '<div data-bbox="1 1 2 2" data-label="Title">\n<p>최종</p>\n</div>'
    )
    elements = parse_chandra_output(raw)
    assert len(elements) == 1
    assert elements[0]["text"] == "최종"


def test_parse_chandra_output_extracts_category_text_and_bbox():
    elements = parse_chandra_output(CHANDRA_SAMPLE)
    cats = [e["category"] for e in elements]
    assert cats == ["Title", "Table", "Page-footer"]
    assert elements[0]["text"] == "문서 제목"
    assert elements[0]["bbox"] == [10.0, 10.0, 500.0, 60.0]
    assert elements[1]["text"] == "<table><tr><td>a</td><td>b</td></tr></table>"


def test_parse_dots_ocr_output_tolerates_surrounding_chatter_and_fences():
    elements = parse_dots_ocr_output(DOTS_SAMPLE)
    assert len(elements) == 2
    assert elements[0] == {"category": "Title", "text": "문서 제목", "bbox": [10, 10, 500, 60]}
    assert elements[1]["category"] == "Table"


def test_parse_dots_ocr_output_returns_empty_on_garbage():
    assert parse_dots_ocr_output("이건 그냥 잡담이지 배열이 아님") == []


def test_normalize_dispatches_by_model_key():
    assert normalize("target", CHANDRA_SAMPLE) == parse_chandra_output(CHANDRA_SAMPLE)
    assert normalize("external_a", DOTS_SAMPLE) == parse_dots_ocr_output(DOTS_SAMPLE)
    assert normalize("external_b", "hello world") == parse_paddle_output("hello world")


def test_parse_paddle_output_wraps_raw_text_as_single_text_block():
    # paddle 은 PaddleOCRVL 파이프라인 밖에서(=우리가 직접 chat-completion 으로) 부르면
    # 구조화된 레이아웃을 못 내고 순수 인식 텍스트만 뱉는다 — 통째로 한 블록 취급.
    elements = parse_paddle_output("  Busan is good  \n2025.3.18  ")
    assert elements == [{"category": "Text", "text": "Busan is good  \n2025.3.18", "bbox": None}]
    assert parse_paddle_output("") == []
    assert parse_paddle_output("   ") == []


def test_full_text_normalizes_table_html_and_plain_text_the_same_way():
    chandra_elements = parse_chandra_output(CHANDRA_SAMPLE)
    paddle_elements = parse_paddle_output("문서 제목 a b 1")
    # 포맷이 완전히 다른 두 출력이라도 full_text 는 둘 다 순수 텍스트로 뽑아준다
    # (교차 모델 비교/gt 비교에서 이 함수를 쓰는 이유).
    assert "<table>" not in full_text(chandra_elements)
    assert "<table>" not in full_text(paddle_elements)
    assert "문서 제목" in full_text(chandra_elements)


def test_plain_text_excludes_tables():
    elements = parse_chandra_output(CHANDRA_SAMPLE)
    text = plain_text(elements)
    assert "문서 제목" in text
    assert "<table>" not in text


def test_table_htmls_extracts_only_tables():
    elements = parse_chandra_output(CHANDRA_SAMPLE)
    htmls = table_htmls(elements)
    assert len(htmls) == 1
    assert htmls[0].startswith("<table>")


# ===== normalize_category (레이블 어휘 표준화, 2026-07-20) =====


def test_normalize_category_keeps_standard_as_is():
    for cat in ("Text", "Table", "Title", "Picture", "Formula"):
        assert normalize_category(cat) == cat


def test_normalize_category_fixes_case_and_punctuation_variants():
    assert normalize_category("list-item") == "List-item"
    assert normalize_category("list_item") == "List-item"
    assert normalize_category("text") == "Text"
    assert normalize_category("section-header") == "Section-header"
    assert normalize_category("page_footer") == "Page-footer"
    assert normalize_category("table") == "Table"
    assert normalize_category("picture") == "Picture"


def test_normalize_category_maps_known_free_form_labels():
    assert normalize_category("reference_content") == "Text"
    assert normalize_category("aside_text") == "Text"
    assert normalize_category("chart") == "Picture"
    assert normalize_category("image") == "Picture"
    assert normalize_category("Page-number") == "Page-footer"
    assert normalize_category("Figure-caption") == "Caption"
    assert normalize_category("display_formula") == "Formula"


def test_normalize_category_falls_back_to_text_for_unknown_long_tail():
    assert normalize_category("IDO1 Library Screen Diagram") != ""  # 뭔가는 반환
    assert normalize_category("some totally novel one-off label") == "Text"
    assert normalize_category("") == "Text"


def test_parse_chandra_output_normalizes_category_and_preserves_lowercase_table_html():
    # 2026-07-20 실데이터에서 발견된 버그 재현: category 정규화 전에 "Table" 과 정확히
    # 문자열 비교하면 소문자 "table" 이 표로 인식 안 돼서 <table> 원본 HTML 이 태그
    # 벗겨진 텍스트로 깨졌었다. 정규화 후 비교해야 이게 안 깨진다.
    raw = (
        '<div data-bbox="1 1 2 2" data-label="table">\n'
        "<table><tr><td>a</td></tr></table>\n"
        "</div>"
    )
    elements = parse_chandra_output(raw)
    assert elements[0]["category"] == "Table"
    assert elements[0]["text"] == "<table><tr><td>a</td></tr></table>"  # 태그 안 벗겨짐


def test_parse_dots_ocr_output_normalizes_category():
    raw = '[{"bbox": [0,0,10,10], "category": "list-item", "text": "a"}]'
    elements = parse_dots_ocr_output(raw)
    assert elements[0]["category"] == "List-item"
