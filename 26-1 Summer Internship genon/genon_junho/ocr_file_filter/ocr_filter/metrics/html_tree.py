"""HTML(표/div-HTML) 문자열 → 트리. teds.py 의 트리 편집거리 계산용 전처리."""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser

_VOID_TAGS = {"br", "img", "hr", "input"}


@dataclass
class Node:
    tag: str
    attrs: dict
    text: str = ""              # 자식 태그로 내려가기 전 직속 텍스트만 (strip 됨)
    children: list["Node"] = field(default_factory=list)

    @property
    def label(self) -> tuple:
        """TEDS 논문 기준 라벨: (tag, colspan, rowspan, text)."""
        colspan = self.attrs.get("colspan", "1")
        rowspan = self.attrs.get("rowspan", "1")
        return (self.tag, colspan, rowspan, self.text)

    @property
    def struct_label(self) -> tuple:
        """TEDS-S: 텍스트 제외, 구조(태그+span)만."""
        colspan = self.attrs.get("colspan", "1")
        rowspan = self.attrs.get("rowspan", "1")
        return (self.tag, colspan, rowspan)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)


class _TreeBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node(tag="__root__", attrs={})
        self._stack: list[Node] = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Node(tag=tag, attrs={k: (v or "") for k, v in attrs})
        self._stack[-1].children.append(node)
        if tag not in _VOID_TAGS:
            self._stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Node(tag=tag, attrs={k: (v or "") for k, v in attrs})
        self._stack[-1].children.append(node)

    def handle_endtag(self, tag: str) -> None:
        # 짝 안 맞는 태그(파싱 실패한 모델 출력 등)는 최선을 다해 무시.
        for i in range(len(self._stack) - 1, 0, -1):
            if self._stack[i].tag == tag:
                del self._stack[i:]
                return

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            cur = self._stack[-1]
            cur.text = (cur.text + " " + text).strip() if cur.text else text


def parse_html(html: str) -> Node:
    """루트(가상 __root__) 아래 실제 최상위 태그들을 자식으로 담아 반환."""
    builder = _TreeBuilder()
    builder.feed(html or "")
    return builder.root
