"""report.diff 단위 테스트: 단어 단위 diff HTML.

    pytest tests/test_report_diff.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_filter.report.diff import word_diff_html  # noqa: E402


def _body(out: str) -> str:
    return out.split('<div class="diff-body">')[1]


def test_identical_text_has_no_del_or_ins():
    out = word_diff_html("hello world", "hello world")
    body = _body(out)
    assert "<del>" not in body
    assert "<ins>" not in body
    assert "hello world" in body


def test_missing_word_marked_as_del():
    out = word_diff_html("hello beautiful world", "hello world")
    assert "<del>beautiful </del>" in out or "<del>beautiful</del>" in out


def test_extra_word_marked_as_ins():
    out = word_diff_html("hello world", "hello extra world")
    assert "<ins>" in out
    assert "extra" in out


def test_both_empty_shows_placeholder():
    out = word_diff_html("", "")
    assert "빈 텍스트" in out


def test_html_is_escaped():
    out = word_diff_html("<script>a</script>", "<script>a</script>")
    assert "<script>a</script>" not in out
    assert "&lt;script&gt;" in out
