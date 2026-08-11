"""통합 HTML 파이프라인 유틸리티.

JSON 파이프라인(`build_unified_dataset.py`의 element 배열)과 동일한 입력을
받아서, 레이아웃/테이블을 **하나의 HTML 문서 조각**으로 변환한다.

설계 원칙
--------
- Datalab Chandra OCR 출력 스타일을 참고하되, 테이블 데이터에는 없는
  `<!DOCTYPE html>`, `<html>`, `<head>`, `<body>` 같은 래퍼 태그는 쓰지 않는다.
- 테이블 데이터가 쓰던 태그(table/thead/tbody/tr/td/th/caption/p/span/br/b/i/u/sup/sub)
  를 우선 사용하고, 레이아웃 전용 카테고리(제목/리스트/그림 등)에 한해 추가 태그
  (h1/h2/h3/ul/li/img)를 허용한다.
- 테이블 element 의 text 는 이미 표 HTML 이므로 그대로 둔다(변형 금지).
- 레이아웃 element 의 text 앞에 붙은 markdown heading prefix(`#`, `##`, ...)는 제거한다.

이 모듈은 빌드(정답 생성)와 평가(필요 시) 양쪽에서 공용으로 사용한다.
"""

from __future__ import annotations

import html as _html_lib
import json
import re
from typing import Any, Optional

# 레이아웃 카테고리 → HTML 태그 매핑.
# 테이블 태그 우선, 그 외 카테고리는 최소한의 추가 태그만 사용한다.
_CATEGORY_TAG = {
    "title": "h1",
    "section-header": "h2",
    "text": "p",
    "list-item": "li",          # 인접한 List-item 은 <ul> 로 묶는다
    "table": "table",           # text 가 이미 표 HTML
    "caption": "p",
    "footnote": "p",
    "formula": "p",
    "picture": "img",           # 내용 없음(자기 닫힘 태그)
    "page-header": "p",
    "page-footer": "p",
}

# markdown heading prefix 제거용 (텍스트 맨 앞의 #, ##, ... ######)
# 주의: bullet(-, *, ○ 등)은 실제 내용일 수 있으므로 제거하지 않고 그대로 보존한다.
_MD_HEADING_RE = re.compile(r"^\s*#{1,6}\s*")


def _strip_markdown_prefix(text: str) -> str:
    """레이아웃 text 앞의 markdown heading prefix(#) 만 제거한다."""
    s = str(text or "")
    s = _MD_HEADING_RE.sub("", s)
    return s.strip()


def _escape_text(text: str) -> str:
    """HTML 텍스트 노드 escape (개행은 <br/> 로)."""
    s = _html_lib.escape(str(text or ""), quote=False)
    s = s.replace("\n", "<br/>")
    return s


def _looks_like_table_html(text: str) -> bool:
    return "<table" in str(text or "").lower()


def layout_elements_to_html(elements: list[dict]) -> str:
    """레이아웃 element 리스트를 HTML 문서 조각으로 변환한다.

    - element 순서를 reading order 로 보고 그대로 유지한다.
    - 인접한 List-item 들은 하나의 <ul> 로 묶는다.
    - Table element 의 text(표 HTML)는 그대로 삽입한다.
    """
    parts: list[str] = []
    open_ul = False

    def close_ul():
        nonlocal open_ul
        if open_ul:
            parts.append("</ul>")
            open_ul = False

    for el in elements:
        if not isinstance(el, dict):
            continue
        category = str(el.get("category", "") or "").strip()
        tag = _CATEGORY_TAG.get(category.lower(), "p")
        raw_text = el.get("text", "") or ""

        if tag == "table" or _looks_like_table_html(raw_text):
            close_ul()
            # 표 HTML 은 변형 없이 그대로 둔다.
            parts.append(str(raw_text).strip())
            continue

        if tag == "li":
            text = _escape_text(_strip_markdown_prefix(raw_text))
            if not open_ul:
                parts.append('<ul style="list-style-type: none">')
                open_ul = True
            parts.append(f"<li>{text}</li>")
            continue

        close_ul()

        if tag == "img":
            # Picture: 내용 텍스트가 있으면 alt 로 보존
            alt = _html_lib.escape(_strip_markdown_prefix(raw_text), quote=True)
            parts.append(f'<img alt="{alt}"/>' if alt else "<img/>")
            continue

        text = _escape_text(_strip_markdown_prefix(raw_text))
        parts.append(f"<{tag}>{text}</{tag}>")

    close_ul()
    return "".join(parts)


def table_element_to_html(text: str) -> str:
    """테이블 element 의 표 HTML 을 그대로 반환한다(HTML 파이프라인에서는 변형 불필요)."""
    return str(text or "").strip()


def unified_json_target_to_html(gt_html_json: str) -> str:
    """JSON 통합 정답 문자열을 HTML 통합 정답 문자열로 변환한다.

    table 단일 element 면 표 HTML 그대로, layout 이면 element → HTML 변환.
    파싱 실패 시 원본을 그대로 반환한다(레거시 HTML 가능성).
    """
    s = str(gt_html_json or "")
    if not s:
        return s
    try:
        elements = json.loads(s)
    except Exception:
        return s
    if not isinstance(elements, list):
        return s
    # 단일 Table element → 표 HTML 그대로
    if (
        len(elements) == 1
        and isinstance(elements[0], dict)
        and str(elements[0].get("category", "")).strip().lower() == "table"
    ):
        return table_element_to_html(elements[0].get("text", ""))
    return layout_elements_to_html(elements)


def extract_html_body(text: str) -> str:
    """모델 출력에서 평가 대상 HTML 본문을 추출한다.

    - <think>...</think> 제거
    - ```html ... ``` 코드펜스 제거
    - <body>...</body> 가 있으면 그 안만, <!DOCTYPE>/<html>/<head> 래퍼는 제거
    """
    s = str(text or "")
    if "</think>" in s:
        s = s.split("</think>")[-1]
    s = s.strip()
    # 코드펜스 제거
    fence = re.search(r"```(?:html)?\s*(.*?)```", s, re.DOTALL | re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    # body 추출
    body = re.search(r"<body[^>]*>(.*?)</body>", s, re.DOTALL | re.IGNORECASE)
    if body:
        return body.group(1).strip()
    # 래퍼 태그 제거(있다면)
    s = re.sub(r"</?(?:!doctype|html|head|body|meta|title)\b[^>]*>", "", s, flags=re.IGNORECASE)
    return s.strip()
