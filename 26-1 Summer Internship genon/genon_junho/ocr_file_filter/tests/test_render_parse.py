"""render-then-verify (MinerU2.5-Pro §3.3) 단위 테스트.

judge 가 보는 이미지에 **파싱 결과를 실제로 렌더한 그림**이 포함돼야 한다. 예전처럼
원본+bbox 오버레이 2장만 주면 표/수식의 내용·구조는 원리적으로 검증할 수 없다.

렌더 백엔드(playwright/pdflatex)가 없는 환경에서도 테스트가 깨지면 안 되므로, 렌더
자체를 요구하는 테스트는 skipif 로 감싸고 나머지는 폴백 동작(렌더 실패 시 조용히 생략)을 본다.

    pytest tests/test_render_parse.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_filter.hardcase.pipeline import judge_layout_once  # noqa: E402
from ocr_filter.hardcase.prompts import build_layout_judge_prompt  # noqa: E402
from ocr_filter.report.render_parse import (  # noqa: E402
    _strip_math,
    render_formulas_latex,
    render_parsed_elements,
    render_tables_html,
)

_HAS_PLAYWRIGHT = True
try:
    import playwright  # noqa: F401
except ImportError:
    _HAS_PLAYWRIGHT = False

_HAS_LATEX = bool(shutil.which("pdflatex") and (shutil.which("magick") or shutil.which("convert")))

TABLE = "<table><tr><td>a</td><td>b</td></tr></table>"


def _el(cat, text, bbox=None):
    return {"category": cat, "text": text, "bbox": bbox}


def _page(tmp: Path) -> str:
    p = tmp / "page.png"
    Image.new("RGB", (800, 1000), "white").save(p)
    return str(p)


# ── 렌더 대상 선별 ────────────────────────────────────────────────────────────
def test_render_parsed_elements_ignores_plain_text():
    """본문 텍스트는 렌더 대상이 아니다 — 글자를 다시 글자로 그릴 뿐 대조 가치가 없다."""
    out = render_parsed_elements([_el("Text", "그냥 본문"), _el("Page-header", "머리말")])
    assert out == {}


def test_render_parsed_elements_skips_table_without_tags():
    """태그가 없으면 표 구조가 아니라 평문이므로 렌더하지 않는다."""
    out = render_parsed_elements([_el("Table", "표인데 태그가 없음")])
    assert "tables" not in out


def test_render_tables_html_empty_returns_none():
    assert render_tables_html([]) is None


def test_render_formulas_latex_empty_returns_none():
    assert render_formulas_latex([]) is None


# ── 실제 렌더 ────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not _HAS_PLAYWRIGHT, reason="playwright 미설치")
def test_render_tables_html_produces_image():
    img = render_tables_html([TABLE])
    assert img is not None
    assert img.size[0] > 0 and img.size[1] > 0


@pytest.mark.skipif(not _HAS_PLAYWRIGHT, reason="playwright 미설치")
def test_broken_table_still_renders_without_raising():
    """닫히지 않은 태그 같은 결함이 있어도 렌더는 되어야 한다 — 그 '깨진 모습' 자체가
    judge 에게 보여주려는 신호이기 때문이다(렌더가 실패해 이미지가 빠지면 안 됨)."""
    img = render_tables_html(["<table><tr><td>깨짐<tr><td>닫힘없음</table>"])
    assert img is not None


@pytest.mark.skipif(not _HAS_LATEX, reason="pdflatex/magick 미설치")
def test_render_formulas_latex_produces_image():
    img = render_formulas_latex([r"E = mc^2"])
    assert img is not None


def test_strip_math_removes_wrappers_only():
    assert _strip_math(r"$E=mc^2$") == "E=mc^2"
    assert _strip_math(r"\(x\)") == "x"
    assert _strip_math("E=mc^2") == "E=mc^2"
    assert _strip_math("$") == "$"  # 껍질만 있는 건 건드리지 않는다


# ── 프롬프트 ─────────────────────────────────────────────────────────────────
def test_prompt_without_renders_is_unchanged():
    base = build_layout_judge_prompt([_el("Text", "x", [0, 0, 10, 10])])
    assert "IMAGE 3" not in base


def test_prompt_numbers_render_images_after_original_and_overlay():
    """IMAGE 번호는 호출측이 넣은 순서(1=원본, 2=오버레이, 3~=렌더)와 맞아야 한다."""
    p = build_layout_judge_prompt([_el("Text", "x", [0, 0, 10, 10])],
                                  rendered_kinds=["tables", "formulas"])
    assert "IMAGE 3: the parsed TABLES" in p
    assert "IMAGE 4: the parsed FORMULAS" in p


def test_prompt_formula_only_starts_at_image_3():
    p = build_layout_judge_prompt([_el("Text", "x", [0, 0, 10, 10])],
                                  rendered_kinds=["formulas"])
    assert "IMAGE 3: the parsed FORMULAS" in p
    assert "IMAGE 4" not in p


# ── judge 연결 ───────────────────────────────────────────────────────────────
def _capture_call(store):
    def fake(cfg, images, prompt, **kw):
        store["images"] = images
        store["prompt"] = prompt
        return '{"pass": true, "overall_score": 5, "issues": []}'
    return fake


def test_judge_sends_only_two_images_when_nothing_to_render():
    with tempfile.TemporaryDirectory() as d:
        store = {}
        judge_layout_once(_page(Path(d)), [_el("Text", "본문", [0, 0, 100, 100])],
                          {"name": "x", "endpoint": "y"}, call_fn=_capture_call(store))
        assert len(store["images"]) == 2  # 원본 + 오버레이
        assert "IMAGE 3" not in store["prompt"]


@pytest.mark.skipif(not _HAS_PLAYWRIGHT, reason="playwright 미설치")
def test_judge_appends_rendered_table_image():
    with tempfile.TemporaryDirectory() as d:
        store = {}
        judge_layout_once(_page(Path(d)), [_el("Table", TABLE, [0, 0, 700, 300])],
                          {"name": "x", "endpoint": "y"}, call_fn=_capture_call(store))
        assert len(store["images"]) == 3
        assert "IMAGE 3: the parsed TABLES" in store["prompt"]


def test_judge_render_content_flag_disables_rendering():
    """비용/시간 때문에 렌더를 끄고 싶을 때 기존 2장 방식으로 되돌아가야 한다."""
    with tempfile.TemporaryDirectory() as d:
        store = {}
        judge_layout_once(_page(Path(d)), [_el("Table", TABLE, [0, 0, 700, 300])],
                          {"name": "x", "endpoint": "y"}, call_fn=_capture_call(store),
                          render_content=False)
        assert len(store["images"]) == 2
