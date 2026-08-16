"""
평가 메트릭

1. TEDS (Tree-Edit-Distance-based Similarity)
   - 전체 테이블 구조 + 내용 유사도
2. TEDS-Structure
   - 셀 내용 제외, 순수 구조만 비교
3. Span Cell F1
   - colspan/rowspan 셀의 정밀도/재현율
4. Span Attribute Accuracy
   - span 값(숫자)의 정확도
5. Simple Table Accuracy
   - 단순 테이블 성능 유지 여부
"""

import re
import math
from collections import Counter, deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional

import editdistance
from apted import APTED, Config
from apted.helpers import Tree
from bs4 import BeautifulSoup
from lxml import etree, html as lxml_html

_TABLE_OPEN_TAG_RE = re.compile(r"<table\b[^>]*>", re.IGNORECASE)
_TABLE_ANY_TAG_RE = re.compile(r"</?table\b[^>]*>", re.IGNORECASE)


def _extract_first_balanced_table_safe(html_text: str) -> str:
    """첫 번째 balanced <table>...</table> 조각을 추출한다.

    utils.html_utils.extract_first_balanced_table가 없는 실행 환경을 위해
    로컬 fallback 구현을 항상 제공한다.
    """
    text = str(html_text or "").strip()
    if not text:
        return ""

    try:
        from utils.html_utils import extract_first_balanced_table as _extract

        result = _extract(text)
        if result:
            return str(result).strip()
    except Exception:
        pass

    start_match = _TABLE_OPEN_TAG_RE.search(text)
    if start_match is None:
        return ""

    start = start_match.start()
    depth = 0
    for tag_match in _TABLE_ANY_TAG_RE.finditer(text, start):
        tag_text = tag_match.group(0).lower()
        is_close = tag_text.startswith("</")
        is_self_closing = tag_text.endswith("/>")

        if not is_close:
            if is_self_closing:
                if depth == 0:
                    return text[start:tag_match.end()].strip()
                continue
            depth += 1
            continue

        if depth <= 0:
            continue
        depth -= 1
        if depth == 0:
            return text[start:tag_match.end()].strip()

    return text[start:].strip()


@lru_cache(maxsize=512)
def _parse_html_table_cached(html: str):
    from utils.html_utils import parse_html_table

    return parse_html_table(html)


def clear_metrics_caches() -> None:
    """평가 중 누적된 메트릭 파싱 캐시를 정리한다."""
    _parse_html_table_cached.cache_clear()


# =============================================================================
# APTED 트리 노드 및 비용 함수 (TFLOP evaluator 기반)
# =============================================================================


class TableTree(Tree):
    """APTED용 테이블 트리 노드.

    Reference: TFLOP/tflop/evaluator.py
    """

    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        if self.tag == "td":
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % (
                self.tag,
                self.colspan,
                self.rowspan,
                self.content,
            )
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class _TEDSConfig(Config):
    """APTED rename 비용 함수. editdistance 사용 (distance 패키지 대체)."""

    def __init__(self, structure_only: bool = False):
        self.structure_only = structure_only

    @staticmethod
    def maximum(*sequences):
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        seq_list = list(sequences)
        max_len = self.maximum(*seq_list)
        if max_len == 0:
            return 0.0
        return float(editdistance.eval(seq_list[0], seq_list[1])) / max_len

    def rename(self, node1, node2):
        """두 노드의 rename 비용. tag/colspan/rowspan 불일치 = 1.0, text = 정규화 편집거리."""
        if (
            (node1.tag != node2.tag)
            or (node1.colspan != node2.colspan)
            or (node1.rowspan != node2.rowspan)
        ):
            return 1.0
        if node1.tag == "td":
            if self.structure_only:
                return 0.0
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.0


# =============================================================================
# TEDS (Tree-Edit-Distance-based Similarity)
# =============================================================================


@dataclass
class TEDSResult:
    """TEDS 계산 결과."""
    teds: float = 0.0
    teds_structure: float = 0.0


class TEDSCalculator:
    """
    TEDS 계산기 (APTED 기반).

    TFLOP evaluator(APTED + 최적 tree edit distance)와 동일한 알고리즘 사용.
    기존 순차 비교 방식 대비 정확한 최적 정렬을 보장한다.

    Reference:
        - https://github.com/ibm-aur-nlp/PubTabNet
        - TFLOP/tflop/evaluator.py
    """

    def __init__(self, structure_only: bool = False, normalize: bool = False):
        self.structure_only = structure_only
        self.normalize = normalize

    def compute(self, pred_html: str, gt_html: str) -> float:
        """두 HTML 테이블의 TEDS 점수를 계산한다 (APTED 기반)."""
        try:
            # normalize=True이면 계산 전에 양쪽 HTML 정규화
            if self.normalize:
                pred_html = normalize_table_structure(pred_html)
                gt_html = normalize_table_structure(gt_html)
            pred_tree, n_nodes_pred = self._html_to_tree(pred_html)
            gt_tree, n_nodes_gt = self._html_to_tree(gt_html)

            if pred_tree is None or gt_tree is None:
                return 0.0

            n_nodes = max(n_nodes_pred, n_nodes_gt)
            if n_nodes == 0:
                return 1.0

            config = _TEDSConfig(structure_only=self.structure_only)
            distance = APTED(pred_tree, gt_tree, config).compute_edit_distance()

            score = 1.0 - (float(distance) / n_nodes)
            if not math.isfinite(score):
                return 0.0
            return max(0.0, min(1.0, score))

        except Exception:
            return 0.0

    def _html_to_tree(self, html_str: str) -> tuple[Optional[TableTree], int]:
        """HTML을 APTED용 TableTree로 변환. (트리, 노드 수) 반환."""
        normalized = self._normalize_table_html(html_str)
        if not normalized:
            return None, 0

        try:
            parser = lxml_html.HTMLParser(remove_comments=True, encoding="utf-8")
            doc = lxml_html.fromstring(normalized, parser=parser)
        except Exception:
            return None, 0

        tables = doc.xpath("body/table")
        if not tables:
            # BeautifulSoup 파싱 폴백
            return self._html_to_tree_bs4(normalized)

        table_elem = tables[0]
        # thead/tbody/tfoot 제거 (이미 정규화에서 제거했지만 lxml이 재생성할 수 있음)
        etree.strip_tags(table_elem, "thead", "tbody", "tfoot")

        n_nodes = len(table_elem.xpath(".//*"))
        tree = self._load_html_tree(table_elem)
        return tree, n_nodes

    def _html_to_tree_bs4(self, normalized_html: str) -> tuple[Optional[TableTree], int]:
        """lxml 실패 시 BeautifulSoup 폴백."""
        try:
            soup = BeautifulSoup(normalized_html, "html.parser")
            table = soup.find("table")
            if table is None:
                return None, 0
            tree, n_nodes = self._bs4_tag_to_table_tree(table)
            return tree, n_nodes
        except Exception:
            return None, 0

    def _load_html_tree(self, node, parent=None) -> Optional[TableTree]:
        """lxml Element를 TableTree로 재귀 변환."""
        if node.tag == "td":
            if self.structure_only:
                cell = []
            else:
                cell = self._tokenize(node)
            new_node = TableTree(
                node.tag,
                int(node.attrib.get("colspan", "1")),
                int(node.attrib.get("rowspan", "1")),
                cell,
                *deque(),
            )
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())

        if parent is not None:
            parent.children.append(new_node)

        if node.tag != "td":
            for child in node:
                self._load_html_tree(child, new_node)

        if parent is None:
            return new_node

    def _tokenize(self, node) -> list[str]:
        """lxml 노드의 텍스트를 토큰 리스트로 변환."""
        tokens = []
        tokens.append("<%s>" % node.tag)
        if node.text is not None:
            text = re.sub(r"\s+", " ", node.text).strip()
            tokens += list(text)
        for child in node:
            tokens += self._tokenize_recursive(child)
        if node.tag != "unk":
            tokens.append("</%s>" % node.tag)
        if node.tag != "td" and node.tail is not None:
            tail = re.sub(r"\s+", " ", node.tail).strip()
            tokens += list(tail)
        # 바깥 <td>...</td> 태그 제거
        return tokens[1:-1] if len(tokens) >= 2 else tokens

    def _tokenize_recursive(self, node) -> list[str]:
        """자식 노드 토큰화 (재귀)."""
        tokens = []
        tokens.append("<%s>" % node.tag)
        if node.text is not None:
            text = re.sub(r"\s+", " ", node.text).strip()
            tokens += list(text)
        for child in node:
            tokens += self._tokenize_recursive(child)
        if node.tag != "unk":
            tokens.append("</%s>" % node.tag)
        if node.tail is not None:
            tail = re.sub(r"\s+", " ", node.tail).strip()
            tokens += list(tail)
        return tokens

    def _bs4_tag_to_table_tree(self, tag) -> tuple[TableTree, int]:
        """BeautifulSoup 태그를 TableTree로 변환 (폴백용)."""
        n_nodes = 0

        if tag.name in ("td", "th"):
            n_nodes += 1
            if self.structure_only:
                cell = []
            else:
                text = tag.get_text(strip=True)
                text = re.sub(r"\s+", " ", text).strip() if text else ""
                cell = list(text)
            return TableTree(
                "td",
                int(tag.get("colspan", 1)),
                int(tag.get("rowspan", 1)),
                cell,
                *deque(),
            ), n_nodes

        node = TableTree(tag.name, None, None, None, *deque())
        n_nodes += 1

        for child in tag.children:
            if isinstance(child, str):
                continue
            if child.name in ("table", "tr", "td", "th"):
                child_tree, child_count = self._bs4_tag_to_table_tree(child)
                node.children.append(child_tree)
                n_nodes += child_count

        return node, n_nodes

    def _normalize_table_html(self, html_str: str) -> str:
        """테이블 HTML 정규화."""
        if not html_str:
            return ""

        cleaned = str(html_str).strip().replace("\x00", "")
        if not cleaned:
            return ""

        # 첫 번째 table 조각만 추출 (nested table 균형 고려)
        fragment = _extract_first_balanced_table_safe(cleaned)
        if not fragment:
            m = re.search(r"<table\b[\s\S]*?</table>", cleaned, flags=re.IGNORECASE)
            fragment = m.group(0) if m else cleaned

        if "<table" not in fragment.lower():
            return ""

        # thead/tbody/tfoot/th → 평탄화
        fragment = re.sub(r"</?thead[^>]*>", "", fragment, flags=re.IGNORECASE)
        fragment = re.sub(r"</?tbody[^>]*>", "", fragment, flags=re.IGNORECASE)
        fragment = re.sub(r"</?tfoot[^>]*>", "", fragment, flags=re.IGNORECASE)
        fragment = re.sub(r"<th\b", "<td", fragment, flags=re.IGNORECASE)
        fragment = re.sub(r"</th>", "</td>", fragment, flags=re.IGNORECASE)
        fragment = re.sub(r"<table[^>]*>", "<table>", fragment, flags=re.IGNORECASE)
        fragment = re.sub(r"<br\s*/?>", " ", fragment, flags=re.IGNORECASE)

        # 빈 <tr></tr> 행 제거 (rowspan continuation으로 생긴 빈 행)
        fragment = re.sub(r"<tr[^>]*>\s*</tr>", "", fragment, flags=re.IGNORECASE)

        return f"<html><body>{fragment}</body></html>"


# =============================================================================
# Generic HTML Tree TEDS (HTML 파이프라인의 레이아웃 평가용)
# =============================================================================
# 테이블 전용 TEDSCalculator 는 <table>/<td> 중심으로만 동작하므로, 페이지 전체를
# 임의 태그 트리(h1/h2/p/ul/li/table 등)로 비교하기 위한 일반 트리 TEDS 를 둔다.
# bbox 가 없는 HTML 출력에서는 IoU 가 불가능하므로 레이아웃도 트리 편집거리 기반
# TEDS 로 평가한다.


class _GenericNode(Tree):
    """임의 HTML 태그 트리 노드 (APTED 용)."""

    def __init__(self, tag, content=None, *children):
        self.tag = tag
        self.content = content if content is not None else []
        self.children = list(children)

    def bracket(self):
        result = '"tag": %s, "text": %s' % (self.tag, self.content)
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class _GenericTEDSConfig(Config):
    """일반 HTML 트리 rename 비용. tag 불일치=1.0, 같으면 텍스트 정규화 편집거리."""

    def __init__(self, structure_only: bool = False):
        self.structure_only = structure_only

    @staticmethod
    def maximum(*sequences):
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        seq = list(sequences)
        max_len = self.maximum(*seq)
        if max_len == 0:
            return 0.0
        return float(editdistance.eval(seq[0], seq[1])) / max_len

    def rename(self, node1, node2):
        if node1.tag != node2.tag:
            return 1.0
        if self.structure_only:
            return 0.0
        if node1.content or node2.content:
            return self.normalized_distance(node1.content, node2.content)
        return 0.0


class GenericHTMLTEDS:
    """임의 HTML 조각(레이아웃 전체 페이지)의 TEDS 를 계산한다."""

    # 비교 대상 태그(그 외 태그는 무시하고 자식만 끌어올린다)
    _ALLOWED_TAGS = {
        "h1", "h2", "h3", "p", "div", "ul", "ol", "li", "img",
        "table", "thead", "tbody", "tfoot", "tr", "td", "th", "caption",
        "span", "b", "i", "u", "sup", "sub",
    }

    def __init__(self, structure_only: bool = False):
        self.structure_only = structure_only

    def compute(self, pred_html: str, gt_html: str) -> float:
        try:
            pred_tree, n_pred = self._html_to_tree(pred_html)
            gt_tree, n_gt = self._html_to_tree(gt_html)
            if pred_tree is None or gt_tree is None:
                return 0.0
            n_nodes = max(n_pred, n_gt)
            if n_nodes == 0:
                return 1.0
            config = _GenericTEDSConfig(structure_only=self.structure_only)
            distance = APTED(pred_tree, gt_tree, config).compute_edit_distance()
            return _to_teds_score(distance, n_nodes)
        except Exception:
            return 0.0

    def _html_to_tree(self, html_str: str):
        s = str(html_str or "").strip().replace("\x00", "")
        if not s:
            return None, 0
        try:
            soup = BeautifulSoup(s, "html.parser")
        except Exception:
            return None, 0
        root = _GenericNode("doc")
        n = self._append_children(soup, root)
        return root, n

    def _append_children(self, parent_soup, parent_node) -> int:
        n_nodes = 0
        for child in getattr(parent_soup, "children", []):
            name = getattr(child, "name", None)
            if name is None:
                continue  # 텍스트 노드는 부모 content 로 별도 처리
            if name not in self._ALLOWED_TAGS:
                # 허용되지 않은 태그는 건너뛰되 자식은 끌어올린다
                n_nodes += self._append_children(child, parent_node)
                continue
            if self.structure_only:
                content = []
            else:
                text = child.get_text(" ", strip=True)
                content = list(re.sub(r"\s+", " ", text).strip()) if text else []
            node = _GenericNode(name, content)
            n_nodes += 1
            n_nodes += self._append_children(child, node)
            parent_node.children.append(node)
        return n_nodes


def compute_generic_html_teds(pred_html: str, gt_html: str) -> tuple[float, float]:
    """일반 HTML 트리 (TEDS, TEDS-Structure) 를 반환한다."""
    teds = GenericHTMLTEDS(structure_only=False).compute(pred_html, gt_html)
    teds_struct = GenericHTMLTEDS(structure_only=True).compute(pred_html, gt_html)
    return teds, teds_struct


_TEDS_TREE_BUILDER = TEDSCalculator(structure_only=False, normalize=False)
_TEDS_CONFIG_TEXT = _TEDSConfig(structure_only=False)
_TEDS_CONFIG_STRUCTURE = _TEDSConfig(structure_only=True)


def _to_teds_score(distance: float, n_nodes: int) -> float:
    if n_nodes <= 0:
        return 1.0
    score = 1.0 - (float(distance) / float(n_nodes))
    if not math.isfinite(score):
        return 0.0
    return max(0.0, min(1.0, score))


def _compute_teds_pair(pred_html: str, gt_html: str) -> tuple[float, float]:
    """한 쌍의 HTML에서 (TEDS, TEDS-Structure)를 함께 계산한다."""
    try:
        pred_tree, n_nodes_pred = _TEDS_TREE_BUILDER._html_to_tree(pred_html)
        gt_tree, n_nodes_gt = _TEDS_TREE_BUILDER._html_to_tree(gt_html)
        if pred_tree is None or gt_tree is None:
            return 0.0, 0.0

        n_nodes = max(n_nodes_pred, n_nodes_gt)
        if n_nodes == 0:
            return 1.0, 1.0

        distance_text = APTED(
            pred_tree, gt_tree, _TEDS_CONFIG_TEXT
        ).compute_edit_distance()
        distance_structure = APTED(
            pred_tree, gt_tree, _TEDS_CONFIG_STRUCTURE
        ).compute_edit_distance()

        return (
            _to_teds_score(distance_text, n_nodes),
            _to_teds_score(distance_structure, n_nodes),
        )
    except Exception:
        return 0.0, 0.0


def compute_teds_variants(pred_html: str, gt_html: str) -> tuple[float, float, float, float]:
    """TEDS 4종을 계산한다.

    Returns:
        (teds, teds_structure, teds_norm, teds_norm_structure)
    """
    teds, teds_structure = _compute_teds_pair(pred_html, gt_html)

    pred_norm = normalize_table_structure(pred_html)
    gt_norm = normalize_table_structure(gt_html)
    teds_norm, teds_norm_structure = _compute_teds_pair(pred_norm, gt_norm)

    return teds, teds_structure, teds_norm, teds_norm_structure


def _extract_parent_and_immediate_child_tables(html_str: str) -> tuple[str, list[str]]:
    """HTML에서 parent table과 immediate child table 목록을 분리한다."""
    if not html_str:
        return "", []

    cleaned = str(html_str).strip()
    if not cleaned:
        return "", []

    fragment = _extract_first_balanced_table_safe(cleaned) or cleaned
    if "<table" not in fragment.lower():
        return "", []

    try:
        soup = BeautifulSoup(fragment, "lxml")
        root = soup.find("table")
        if root is None:
            return "", []

        child_tables = []
        for child in root.find_all("table"):
            parent_table = child.find_parent("table")
            if parent_table is root:
                child_tables.append(child)

        children_html = [str(child) for child in child_tables]
        for idx, child in enumerate(child_tables, start=1):
            child.replace_with(f"[[NESTED_CHILD_{idx}]]")

        return str(root), children_html
    except Exception:
        return fragment, []


def _average_teds_variant_components(
    component_scores: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    if not component_scores:
        return 0.0, 0.0, 0.0, 0.0
    n = float(len(component_scores))
    return (
        sum(s[0] for s in component_scores) / n,
        sum(s[1] for s in component_scores) / n,
        sum(s[2] for s in component_scores) / n,
        sum(s[3] for s in component_scores) / n,
    )


def _combine_outer_inner_teds(
    outer_scores: tuple[float, float, float, float],
    inner_scores: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """outer/inner 점수를 1:1로 결합한다."""
    return (
        (outer_scores[0] + inner_scores[0]) / 2.0,
        (outer_scores[1] + inner_scores[1]) / 2.0,
        (outer_scores[2] + inner_scores[2]) / 2.0,
        (outer_scores[3] + inner_scores[3]) / 2.0,
    )


def compute_teds_variants_nested_split(
    pred_html: str,
    gt_html: str,
) -> tuple[float, float, float, float]:
    """Nested table을 parent/child로 분리해 TEDS 4종 평균을 계산한다.

    계산식:
      - outer(parent) table TEDS 4종 1개
      - inner(child) table 쌍별 TEDS 4종 평균 (GT child 기준 index 매칭)
      - 최종 점수 = (outer + inner_avg) / 2
    """
    pred_parent, pred_children = _extract_parent_and_immediate_child_tables(pred_html)
    gt_parent, gt_children = _extract_parent_and_immediate_child_tables(gt_html)

    # parent 분리가 안 되는 비정상 HTML은 기존 계산으로 폴백한다.
    if not pred_parent or not gt_parent:
        return compute_teds_variants(pred_html, gt_html)

    outer_scores = compute_teds_variants(pred_parent, gt_parent)
    if not pred_children and not gt_children:
        return outer_scores

    # GT child를 기준으로 inner score를 계산한다.
    # (pred child가 부족한 경우 0점, extra pred child는 무시)
    inner_components: list[tuple[float, float, float, float]] = []
    for idx, gt_child in enumerate(gt_children):
        pred_child = pred_children[idx] if idx < len(pred_children) else ""
        if pred_child and gt_child:
            inner_components.append(compute_teds_variants(pred_child, gt_child))
        else:
            # 한쪽 child가 누락되면 0점 처리해 구조 누락을 강하게 반영한다.
            inner_components.append((0.0, 0.0, 0.0, 0.0))

    if not inner_components:
        inner_scores = (0.0, 0.0, 0.0, 0.0)
    else:
        inner_scores = _average_teds_variant_components(inner_components)

    return _combine_outer_inner_teds(outer_scores, inner_scores)


# =============================================================================
# Span Metrics
# =============================================================================


@dataclass
class SpanCell:
    """평가용 span 셀 표현."""
    row: int
    col: int
    rowspan: int
    colspan: int

    def __hash__(self):
        return hash((self.row, self.col, self.rowspan, self.colspan))

    def __eq__(self, other):
        return (
            self.row == other.row
            and self.col == other.col
            and self.rowspan == other.rowspan
            and self.colspan == other.colspan
        )

    @property
    def position_key(self) -> tuple:
        """위치만으로 비교 (span 크기 무시)."""
        return (self.row, self.col)


@dataclass
class SpanMetrics:
    """Span 관련 메트릭 모음."""
    # Span Cell F1 (위치 + 크기 모두 일치)
    span_precision: float = 0.0
    span_recall: float = 0.0
    span_f1: float = 0.0

    # Span Position F1 (위치만 일치, 크기 무관)
    position_precision: float = 0.0
    position_recall: float = 0.0
    position_f1: float = 0.0

    # Span Attribute Accuracy (위치가 맞는 span에서 크기 정확도)
    attribute_accuracy: float = 0.0

    # colspan/rowspan 별도 정확도
    colspan_accuracy: float = 0.0
    rowspan_accuracy: float = 0.0


def extract_span_cells(html: str) -> list[SpanCell]:
    """HTML에서 span 셀 목록을 추출한다."""
    try:
        structure = _parse_html_table_cached(html)
    except Exception:
        return []

    spans = []
    for cell in structure.cells:
        if cell.colspan > 1 or cell.rowspan > 1:
            spans.append(
                SpanCell(
                    row=cell.row,
                    col=cell.col,
                    rowspan=cell.rowspan,
                    colspan=cell.colspan,
                )
            )
    return spans


def compute_span_metrics(pred_html: str, gt_html: str) -> SpanMetrics:
    """
    예측과 정답 HTML의 span 메트릭을 계산한다.
    """
    pred_spans = extract_span_cells(pred_html)
    gt_spans = extract_span_cells(gt_html)

    metrics = SpanMetrics()

    if not gt_spans and not pred_spans:
        # 둘 다 span이 없으면 완벽
        metrics.span_precision = 1.0
        metrics.span_recall = 1.0
        metrics.span_f1 = 1.0
        metrics.position_precision = 1.0
        metrics.position_recall = 1.0
        metrics.position_f1 = 1.0
        metrics.attribute_accuracy = 1.0
        metrics.colspan_accuracy = 1.0
        metrics.rowspan_accuracy = 1.0
        return metrics

    # --- Span Cell F1 (위치 + 크기 완전 일치) ---
    pred_set = set(pred_spans)
    gt_set = set(gt_spans)

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    metrics.span_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics.span_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics.span_f1 = _f1(metrics.span_precision, metrics.span_recall)

    # --- Position F1 (위치만 일치) ---
    pred_positions = {s.position_key for s in pred_spans}
    gt_positions = {s.position_key for s in gt_spans}

    pos_tp = len(pred_positions & gt_positions)
    pos_fp = len(pred_positions - gt_positions)
    pos_fn = len(gt_positions - pred_positions)

    metrics.position_precision = pos_tp / (pos_tp + pos_fp) if (pos_tp + pos_fp) > 0 else 0.0
    metrics.position_recall = pos_tp / (pos_tp + pos_fn) if (pos_tp + pos_fn) > 0 else 0.0
    metrics.position_f1 = _f1(metrics.position_precision, metrics.position_recall)

    # --- Attribute Accuracy (위치가 맞는 span에서 크기 정확도) ---
    pred_by_pos = {s.position_key: s for s in pred_spans}
    gt_by_pos = {s.position_key: s for s in gt_spans}

    matched_positions = pred_positions & gt_positions
    if matched_positions:
        correct_attrs = 0
        correct_colspan = 0
        correct_rowspan = 0
        total_matched = len(matched_positions)

        for pos in matched_positions:
            p = pred_by_pos[pos]
            g = gt_by_pos[pos]
            if p.colspan == g.colspan and p.rowspan == g.rowspan:
                correct_attrs += 1
            if p.colspan == g.colspan:
                correct_colspan += 1
            if p.rowspan == g.rowspan:
                correct_rowspan += 1

        metrics.attribute_accuracy = correct_attrs / total_matched
        metrics.colspan_accuracy = correct_colspan / total_matched
        metrics.rowspan_accuracy = correct_rowspan / total_matched

    return metrics


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# =============================================================================
# Grid Metrics (GCA / GSA)
# =============================================================================


@dataclass
class GridMetrics:
    """확장 그리드 기반 보조 메트릭."""
    # GCA (Grid Cell Accuracy): 셀 내용 일치율
    gca: float = 0.0
    # GSA (Grid Structure Accuracy): 인접 셀 병합 관계 일치율
    gsa: float = 0.0


def html_to_expanded_grid(html_str: str) -> list[list[str]]:
    """HTML 테이블을 확장 그리드(2D 배열)로 변환.

    colspan/rowspan을 풀어서 각 물리 셀에 해당 논리 셀의 텍스트를 채운다.
    """
    try:
        structure = _parse_html_table_cached(html_str)
    except Exception:
        return []

    if structure.num_rows == 0 or structure.num_cols == 0:
        return []

    grid = [["" for _ in range(structure.num_cols)] for _ in range(structure.num_rows)]

    for cell in structure.cells:
        text = re.sub(r"\s+", " ", cell.content).strip() if cell.content else ""
        for r in range(cell.row, min(cell.row + cell.rowspan, structure.num_rows)):
            for c in range(cell.col, min(cell.col + cell.colspan, structure.num_cols)):
                grid[r][c] = text

    return grid


def _build_merge_map(html_str: str) -> dict[tuple[int, int], tuple[int, int]]:
    """각 물리 셀 → 속한 논리 셀 원점(row, col)의 매핑을 생성."""
    try:
        structure = _parse_html_table_cached(html_str)
    except Exception:
        return {}

    merge_map: dict[tuple[int, int], tuple[int, int]] = {}
    for cell in structure.cells:
        origin = (cell.row, cell.col)
        for r in range(cell.row, min(cell.row + cell.rowspan, structure.num_rows)):
            for c in range(cell.col, min(cell.col + cell.colspan, structure.num_cols)):
                merge_map[(r, c)] = origin

    return merge_map


def compute_grid_metrics(pred_html: str, gt_html: str) -> GridMetrics:
    """확장 그리드 기반 GCA / GSA를 계산한다."""
    pred_grid = html_to_expanded_grid(pred_html)
    gt_grid = html_to_expanded_grid(gt_html)

    if not pred_grid or not gt_grid:
        return GridMetrics()

    # 공통 크기로 자름 (크기 차이 자체는 별도 페널티 없음)
    rows = min(len(pred_grid), len(gt_grid))
    cols = min(len(pred_grid[0]), len(gt_grid[0]))

    if rows == 0 or cols == 0:
        return GridMetrics()

    # --- GCA: 셀 내용 일치율 ---
    match_count = 0
    total_cells = rows * cols
    for r in range(rows):
        for c in range(cols):
            if pred_grid[r][c] == gt_grid[r][c]:
                match_count += 1
    gca = match_count / total_cells

    # --- GSA: 인접 셀의 병합 관계 일치율 ---
    pred_merge = _build_merge_map(pred_html)
    gt_merge = _build_merge_map(gt_html)

    total_pairs = 0
    merge_match = 0

    for r in range(rows):
        for c in range(cols):
            # 오른쪽 인접
            if c + 1 < cols:
                total_pairs += 1
                pred_same = pred_merge.get((r, c)) == pred_merge.get((r, c + 1))
                gt_same = gt_merge.get((r, c)) == gt_merge.get((r, c + 1))
                if pred_same == gt_same:
                    merge_match += 1
            # 아래쪽 인접
            if r + 1 < rows:
                total_pairs += 1
                pred_same = pred_merge.get((r, c)) == pred_merge.get((r + 1, c))
                gt_same = gt_merge.get((r, c)) == gt_merge.get((r + 1, c))
                if pred_same == gt_same:
                    merge_match += 1

    gsa = merge_match / total_pairs if total_pairs > 0 else 0.0

    return GridMetrics(gca=gca, gsa=gsa)


# =============================================================================
# Normalized Table Structure (지그재그 rowspan 등 비정상 인코딩 정규화)
# =============================================================================


def normalize_table_structure(html_str: str) -> str:
    """HTML 테이블을 확장 그리드로 풀어서 최소 병합으로 재구축한다.

    동일한 시각적 테이블은 항상 동일한 HTML 구조를 갖게 된다.
    지그재그 rowspan=2 등 비정상 인코딩을 정규화한다.

    알고리즘:
      1. parse_html_table() → TableStructure
      2. 확장 그리드 구축: grid[r][c] = cell_id
      3. 중복 행/열 제거 (전체가 윗행/왼쪽열의 연속인 경우)
      4. Greedy 병합: 좌상단부터 스캔하며 동일 cell_id 인접 셀을 병합
      5. structure_to_html()로 재구축
    """
    from utils.html_utils import structure_to_html, CellInfo, TableStructure

    try:
        structure = _parse_html_table_cached(html_str)
    except Exception:
        return html_str  # 파싱 실패 시 원본 반환

    if not structure.cells or structure.num_rows == 0 or structure.num_cols == 0:
        return html_str

    rows = structure.num_rows
    cols = structure.num_cols

    # 1. 확장 그리드: grid[r][c] = cell_id (cells 리스트의 인덱스)
    grid = [[-1] * cols for _ in range(rows)]
    cell_contents: dict[int, str] = {}

    for cell_id, cell in enumerate(structure.cells):
        cell_contents[cell_id] = cell.content
        for r in range(cell.row, min(cell.row + cell.rowspan, rows)):
            for c in range(cell.col, min(cell.col + cell.colspan, cols)):
                grid[r][c] = cell_id

    # 2. 중복 행 제거 (모든 셀이 윗 행과 동일한 cell_id인 행 = continuation row)
    keep_rows = [0]
    for r in range(1, rows):
        is_continuation = all(grid[r][c] == grid[r - 1][c] for c in range(cols))
        if not is_continuation:
            keep_rows.append(r)

    # 3. 중복 열 제거 (모든 셀이 왼쪽 열과 동일한 cell_id인 열)
    keep_cols = [0]
    for c in range(1, cols):
        is_continuation = all(grid[r][c] == grid[r][c - 1] for r in range(rows))
        if not is_continuation:
            keep_cols.append(c)

    new_rows = len(keep_rows)
    new_cols = len(keep_cols)

    if new_rows == 0 or new_cols == 0:
        return html_str

    # 4. 축소된 그리드 생성
    new_grid = [
        [grid[keep_rows[r]][keep_cols[c]] for c in range(new_cols)]
        for r in range(new_rows)
    ]

    # 5. Greedy 병합: 좌상단부터 스캔하며 새 셀 목록 생성
    visited = [[False] * new_cols for _ in range(new_rows)]
    new_cells = []

    for r in range(new_rows):
        for c in range(new_cols):
            if visited[r][c]:
                continue

            cell_id = new_grid[r][c]
            if cell_id < 0:
                visited[r][c] = True
                continue

            # colspan 확장: 오른쪽으로 동일 cell_id 탐색
            max_c = c
            while max_c + 1 < new_cols and new_grid[r][max_c + 1] == cell_id:
                max_c += 1

            # rowspan 확장: 아래로 (colspan 범위 전체가 동일해야 함)
            max_r = r
            while max_r + 1 < new_rows:
                all_same = all(
                    new_grid[max_r + 1][cc] == cell_id
                    for cc in range(c, max_c + 1)
                )
                if all_same:
                    max_r += 1
                else:
                    break

            # visited 마킹
            for rr in range(r, max_r + 1):
                for cc in range(c, max_c + 1):
                    visited[rr][cc] = True

            new_cells.append(CellInfo(
                row=r,
                col=c,
                rowspan=max_r - r + 1,
                colspan=max_c - c + 1,
                content=cell_contents.get(cell_id, ""),
                is_header=False,
            ))

    # 6. 새 TableStructure → HTML
    new_structure = TableStructure(
        num_rows=new_rows,
        num_cols=new_cols,
        cells=new_cells,
        header_rows=0,
    )

    return structure_to_html(new_structure, include_content=True)


# =============================================================================
# Aggregate Metrics
# =============================================================================


@dataclass
class AggregateMetrics:
    """전체 데이터셋의 집계 메트릭."""
    # TEDS
    avg_teds: float = 0.0
    avg_teds_structure: float = 0.0

    # Normalized TEDS (지그재그 rowspan 등 비정상 인코딩 정규화 후 TEDS)
    avg_teds_norm: float = 0.0
    avg_teds_norm_structure: float = 0.0

    # Span metrics
    avg_span_f1: float = 0.0
    avg_span_precision: float = 0.0
    avg_span_recall: float = 0.0
    avg_position_f1: float = 0.0
    avg_attribute_accuracy: float = 0.0
    avg_colspan_accuracy: float = 0.0
    avg_rowspan_accuracy: float = 0.0

    # Grid metrics (보조)
    avg_gca: float = 0.0
    avg_gsa: float = 0.0

    # By complexity
    simple_teds: float = 0.0
    medium_teds: float = 0.0
    complex_teds: float = 0.0

    # Counts
    total_samples: int = 0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


BASE_COMPLEXITY_BUCKETS = ("simple", "medium", "complex")


def normalize_complexity_label(value: Any) -> str:
    """복잡도 라벨을 소문자 문자열로 정규화한다."""
    comp = str(value or "").strip().lower()
    if not comp:
        return "unknown"
    alias_map = {
        "nested": "complex_nested",
        "complexnested": "complex_nested",
        "complex-nested": "complex_nested",
        "complex-col": "complex_col",
        "complexcol": "complex_col",
        "complex-row": "complex_row",
        "complexrow": "complex_row",
        "complex-mix": "complex_mix",
        "complexmix": "complex_mix",
    }
    return alias_map.get(comp, comp)


def to_base_complexity_bucket(value: Any) -> str:
    """세분화 라벨을 기본 버킷(simple/medium/complex)으로 매핑한다."""
    comp = normalize_complexity_label(value)
    if comp in BASE_COMPLEXITY_BUCKETS:
        return comp
    if comp.startswith("complex"):
        return "complex"
    return "unknown"


def compute_aggregate_metrics(
    predictions: list[dict],
) -> AggregateMetrics:
    """
    전체 예측 결과의 집계 메트릭을 계산한다.

    Args:
        predictions: [{"pred_html": str, "gt_html": str, "complexity": str}, ...]

    Returns:
        AggregateMetrics
    """
    teds_calc = TEDSCalculator(structure_only=False)
    teds_struct_calc = TEDSCalculator(structure_only=True)

    all_teds = []
    all_teds_struct = []
    all_teds_norm = []
    all_teds_norm_struct = []
    all_span_f1 = []
    all_span_precision = []
    all_span_recall = []
    all_position_f1 = []
    all_attribute_accuracy = []
    all_colspan_accuracy = []
    all_rowspan_accuracy = []
    all_gca = []
    all_gsa = []

    by_complexity = {k: [] for k in BASE_COMPLEXITY_BUCKETS}

    for item in predictions:
        pred = item.get("pred_html", "")
        gt = item.get("gt_html", "")
        complexity = normalize_complexity_label(item.get("complexity", "unknown"))
        base_complexity = to_base_complexity_bucket(complexity)

        # TEDS: pre-computed 값 재사용, 없으면 계산
        if "teds" in item and "teds_structure" in item:
            teds = float(item["teds"])
            teds_struct = float(item["teds_structure"])
        else:
            teds = teds_calc.compute(pred, gt)
            teds_struct = teds_struct_calc.compute(pred, gt)
        all_teds.append(teds)
        all_teds_struct.append(teds_struct)

        # Normalized TEDS: pre-computed 값 재사용, 없으면 계산
        if "teds_norm" in item:
            all_teds_norm.append(float(item["teds_norm"]))
            all_teds_norm_struct.append(float(item.get("teds_norm_structure", 0.0)))
        else:
            norm_pred = normalize_table_structure(pred)
            norm_gt = normalize_table_structure(gt)
            all_teds_norm.append(teds_calc.compute(norm_pred, norm_gt))
            all_teds_norm_struct.append(teds_struct_calc.compute(norm_pred, norm_gt))

        if base_complexity in by_complexity:
            by_complexity[base_complexity].append(teds)

        # Span metrics: pre-computed 값 재사용, 없으면 계산 (폴백)
        if "span_f1" in item:
            all_span_f1.append(float(item["span_f1"]))
            all_span_precision.append(float(item.get("span_precision", 0.0)))
            all_span_recall.append(float(item.get("span_recall", 0.0)))
            all_position_f1.append(float(item.get("position_f1", 0.0)))
            all_attribute_accuracy.append(float(item.get("attribute_accuracy", 0.0)))
            if "colspan_accuracy" in item and "rowspan_accuracy" in item:
                all_colspan_accuracy.append(float(item.get("colspan_accuracy", 0.0)))
                all_rowspan_accuracy.append(float(item.get("rowspan_accuracy", 0.0)))
            else:
                # Backward compatibility: old predictions may not include these fields.
                span_m = compute_span_metrics(pred, gt)
                all_colspan_accuracy.append(span_m.colspan_accuracy)
                all_rowspan_accuracy.append(span_m.rowspan_accuracy)
        else:
            span_m = compute_span_metrics(pred, gt)
            all_span_f1.append(span_m.span_f1)
            all_span_precision.append(span_m.span_precision)
            all_span_recall.append(span_m.span_recall)
            all_position_f1.append(span_m.position_f1)
            all_attribute_accuracy.append(span_m.attribute_accuracy)
            all_colspan_accuracy.append(span_m.colspan_accuracy)
            all_rowspan_accuracy.append(span_m.rowspan_accuracy)

        # Grid metrics: pre-computed 값 재사용, 없으면 계산 (폴백)
        if "gca" in item:
            all_gca.append(float(item["gca"]))
            all_gsa.append(float(item.get("gsa", 0.0)))
        else:
            grid_m = compute_grid_metrics(pred, gt)
            all_gca.append(grid_m.gca)
            all_gsa.append(grid_m.gsa)

    n = len(predictions) or 1

    agg = AggregateMetrics(
        total_samples=len(predictions),
        avg_teds=sum(all_teds) / n,
        avg_teds_structure=sum(all_teds_struct) / n,
        avg_teds_norm=sum(all_teds_norm) / n,
        avg_teds_norm_structure=sum(all_teds_norm_struct) / n,
        avg_span_f1=sum(all_span_f1) / n,
        avg_span_precision=sum(all_span_precision) / n,
        avg_span_recall=sum(all_span_recall) / n,
        avg_position_f1=sum(all_position_f1) / n,
        avg_attribute_accuracy=sum(all_attribute_accuracy) / n,
        avg_colspan_accuracy=sum(all_colspan_accuracy) / n,
        avg_rowspan_accuracy=sum(all_rowspan_accuracy) / n,
        avg_gca=sum(all_gca) / n,
        avg_gsa=sum(all_gsa) / n,
    )

    # By complexity
    for comp in BASE_COMPLEXITY_BUCKETS:
        scores = by_complexity[comp]
        avg = sum(scores) / len(scores) if scores else 0.0
        setattr(agg, f"{comp}_teds", avg)

    return agg
