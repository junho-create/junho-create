"""텍스트 단어 단위 diff → HTML. gt_score/agreement_score 가 왜 그렇게 나왔는지
숫자만으론 안 보이니, 실제로 뭐가 빠지고/틀리고/더 붙었는지 눈으로 보여주려고 만듦.
"""

from __future__ import annotations

import difflib
import html
import re

_WORD_RE = re.compile(r"\S+|\s+")


def word_diff_html(a: str, b: str, a_label: str = "GT", b_label: str = "예측") -> str:
    """a(기준, 보통 GT) vs b(비교대상, 보통 target 예측)의 단어단위 diff.
    빨강 취소선 = a 에만 있음(빠짐), 초록 = b 에만 있음(더 생김/틀림), 나머지 = 공통."""
    a_tokens = _WORD_RE.findall(a or "")
    b_tokens = _WORD_RE.findall(b or "")
    sm = difflib.SequenceMatcher(None, a_tokens, b_tokens, autojunk=False)

    parts = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            parts.append(html.escape("".join(a_tokens[i1:i2])))
        elif op == "delete":
            parts.append(f'<del>{html.escape("".join(a_tokens[i1:i2]))}</del>')
        elif op == "insert":
            parts.append(f'<ins>{html.escape("".join(b_tokens[j1:j2]))}</ins>')
        elif op == "replace":
            parts.append(f'<del>{html.escape("".join(a_tokens[i1:i2]))}</del>')
            parts.append(f'<ins>{html.escape("".join(b_tokens[j1:j2]))}</ins>')

    body = "".join(parts) if parts else "(둘 다 빈 텍스트)"
    return (
        f'<div class="diff"><div class="diff-label">{html.escape(a_label)} → '
        f'{html.escape(b_label)} 단어 diff (<del>빨강=GT에만 있음(빠짐)</del> '
        f'<ins>초록=예측에만 있음(오탐/오류)</ins>)</div>'
        f'<div class="diff-body">{body}</div></div>'
    )


DIFF_CSS = """
.diff { margin-top: 10px; padding: 8px 10px; background: #151515; border-radius: 6px;
        font-size: 12px; border: 1px solid #2a2a2a; }
.diff-label { color: #888; margin-bottom: 4px; font-size: 11px; }
.diff-body { white-space: pre-wrap; word-break: break-all; line-height: 1.6; }
.diff-body del { color: #ff6b6b; background: rgba(255,107,107,0.12); text-decoration: line-through; }
.diff-body ins { color: #51cf66; background: rgba(81,207,102,0.12); text-decoration: none; }
"""
