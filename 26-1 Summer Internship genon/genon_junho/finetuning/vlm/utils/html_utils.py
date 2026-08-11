"""
HTML 파싱/생성 유틸리티
- 테이블 HTML ↔ 구조 데이터 변환
- TEDS 계산을 위한 HTML 트리 파싱
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional
from bs4 import BeautifulSoup, Tag

_TABLE_OPEN_TAG_RE = re.compile(r"<table\b[^>]*>", re.IGNORECASE)
_TABLE_ANY_TAG_RE = re.compile(r"</?table\b[^>]*>", re.IGNORECASE)


def _safe_positive_int(value: Any, default: int = 1) -> int:
    """span attribute를 안전하게 양의 정수로 변환한다."""
    try:
        parsed = int(str(value).strip())
    except Exception:
        return default
    return parsed if parsed > 0 else default


@dataclass
class CellInfo:
    """테이블 셀 정보."""
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    content: str = ""
    is_header: bool = False

    @property
    def row_end(self) -> int:
        return self.row + self.rowspan - 1

    @property
    def col_end(self) -> int:
        return self.col + self.colspan - 1

    @property
    def is_span(self) -> bool:
        return self.rowspan > 1 or self.colspan > 1


@dataclass
class TableStructure:
    """파싱된 테이블 구조."""
    num_rows: int = 0
    num_cols: int = 0
    cells: list[CellInfo] = field(default_factory=list)
    header_rows: int = 0
    nested_table_count: int = 0  # 중첩 table 태그 수

    @property
    def span_cells(self) -> list[CellInfo]:
        return [c for c in self.cells if c.is_span]

    @property
    def colspan_cells(self) -> list[CellInfo]:
        return [c for c in self.cells if c.colspan > 1]

    @property
    def rowspan_cells(self) -> list[CellInfo]:
        return [c for c in self.cells if c.rowspan > 1]

    @property
    def span_pattern(self) -> str:
        """complex 내 span 패턴 분류: col_only / row_only / mixed / none."""
        n = len(self.span_cells)
        if n < 3:
            return "none"  # simple/medium은 패턴 세분화 불필요
        has_col = len(self.colspan_cells) > 0
        has_row = len(self.rowspan_cells) > 0
        if has_col and has_row:
            return "mixed"
        return "col_only" if has_col else "row_only"

    @property
    def grid_irregularity(self) -> float:
        """그리드 불규칙도 (0.0=완전규칙 ~ 1.0=완전불규칙)."""
        total = self.num_rows * self.num_cols
        if total == 0:
            return 0.0
        return 1.0 - (len(self.cells) / total)

    @property
    def span_complexity(self) -> str:
        """span 복잡도 분류: simple / medium / complex_nested / complex_col / complex_row / complex_mix."""
        n_spans = len(self.span_cells)
        if n_spans == 0:
            return "simple"
        elif n_spans <= 2:
            return "medium"
        # 중첩 테이블 우선 분류
        if self.nested_table_count > 0:
            return "complex_nested"
        # span_pattern으로 세분화
        pat = self.span_pattern
        if pat == "col_only":
            return "complex_col"
        elif pat == "row_only":
            return "complex_row"
        else:
            return "complex_mix"

    @property
    def complexity_category(self) -> str:
        """하위 호환: complex_* → 'complex'로 반환."""
        c = self.span_complexity
        return "complex" if c.startswith("complex") else c


def extract_first_balanced_table(html: str) -> str:
    """문자열에서 첫 번째 <table>...</table> 조각(중첩 포함)을 추출한다."""
    if not html:
        return ""

    text = str(html).strip()
    if not text:
        return ""

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

    # 닫힘 태그가 누락된 비정상 응답이면 table 시작점 이후를 반환한다.
    return text[start:].strip()


def annotate_empty_cells_with_token(
    html: str,
    empty_token: str = "__EMPTY__",
) -> str:
    """빈 td/th 셀을 지정 토큰으로 치환한 table HTML을 반환한다.

    빈 셀 정의:
    - cell.get_text(strip=True)가 비어 있음 (NBSP 포함)
    - nested table을 직접 포함하지 않음
    """
    fragment = extract_first_balanced_table(html) or str(html or "").strip()
    if not fragment:
        return ""

    token = str(empty_token or "").strip()
    if not token:
        return fragment

    try:
        soup = BeautifulSoup(fragment, "lxml")
        table = soup.find("table")
        if table is None:
            return fragment

        changed = False
        for cell in table.find_all(["td", "th"]):
            if cell.find("table") is not None:
                continue
            text = cell.get_text(separator="", strip=True).replace("\xa0", "").strip()
            if text:
                continue
            cell.clear()
            cell.append(token)
            changed = True

        return str(table) if changed else fragment
    except Exception:
        return fragment


def parse_html_table(html: str) -> TableStructure:
    """
    HTML 테이블 문자열을 파싱하여 TableStructure를 반환한다.

    Args:
        html: <table>...</table> HTML 문자열

    Returns:
        TableStructure 객체
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if table is None:
        raise ValueError("No <table> element found in HTML")

    structure = TableStructure()
    # 중첩 테이블 감지: 최상위 <table> 내부의 <table> 태그 수
    structure.nested_table_count = len(table.find_all("table"))
    cells = []

    # grid를 만들어서 span이 차지하는 영역 추적
    grid: dict[tuple[int, int], bool] = {}

    # thead 행 수 계산
    thead = table.find("thead", recursive=False)
    header_row_count = 0
    if thead:
        header_row_count = len(thead.find_all("tr", recursive=False))

    # 중요: nested table 내부 tr/td가 외곽 테이블 파싱에 섞이지 않도록
    # 최상위 section(thead/tbody/tfoot) 또는 table 직계 tr만 순회한다.
    rows: list[Tag] = []
    sections = table.find_all(["thead", "tbody", "tfoot"], recursive=False)
    if sections:
        for section in sections:
            rows.extend(section.find_all("tr", recursive=False))
    else:
        rows = table.find_all("tr", recursive=False)

    row_idx = 0
    for tr in rows:
        col_idx = 0
        is_header_row = row_idx < header_row_count

        for cell_tag in tr.find_all(["td", "th"], recursive=False):
            # 이미 span으로 차지된 열은 건너뜀
            while (row_idx, col_idx) in grid:
                col_idx += 1

            rowspan = _safe_positive_int(cell_tag.get("rowspan", 1), default=1)
            colspan = _safe_positive_int(cell_tag.get("colspan", 1), default=1)
            content = cell_tag.get_text(strip=True)

            cell = CellInfo(
                row=row_idx,
                col=col_idx,
                rowspan=rowspan,
                colspan=colspan,
                content=content,
                is_header=(cell_tag.name == "th" or is_header_row),
            )
            cells.append(cell)

            # grid에 차지된 영역 마킹
            for r in range(row_idx, row_idx + rowspan):
                for c in range(col_idx, col_idx + colspan):
                    grid[(r, c)] = True

            col_idx += colspan

        row_idx += 1

    # 전체 크기 계산
    structure.num_rows = row_idx
    structure.num_cols = max((c + 1 for _, c in grid), default=0)
    structure.cells = cells
    structure.header_rows = header_row_count

    return structure


def structure_to_html(structure: TableStructure, include_content: bool = True) -> str:
    """
    TableStructure를 HTML 문자열로 변환한다.

    Args:
        structure: TableStructure 객체
        include_content: 셀 내용 포함 여부 (False면 빈 셀)

    Returns:
        HTML 문자열
    """
    lines = ["<table>"]

    # 행별로 셀 그룹화
    rows: dict[int, list[CellInfo]] = {}
    for cell in structure.cells:
        rows.setdefault(cell.row, []).append(cell)

    # 각 행 정렬
    for r in rows:
        rows[r].sort(key=lambda c: c.col)

    in_thead = False
    in_tbody = False

    for r in range(structure.num_rows):
        if r < structure.header_rows and not in_thead:
            lines.append("  <thead>")
            in_thead = True
        elif r >= structure.header_rows and in_thead:
            lines.append("  </thead>")
            in_thead = False
        if r >= structure.header_rows and not in_tbody:
            lines.append("  <tbody>")
            in_tbody = True

        lines.append("    <tr>")
        for cell in rows.get(r, []):
            tag = "th" if cell.is_header else "td"
            attrs = ""
            if cell.colspan > 1:
                attrs += f' colspan="{cell.colspan}"'
            if cell.rowspan > 1:
                attrs += f' rowspan="{cell.rowspan}"'

            content = cell.content if include_content else ""
            lines.append(f"      <{tag}{attrs}>{content}</{tag}>")
        lines.append("    </tr>")

    if in_thead:
        lines.append("  </thead>")
    if in_tbody:
        lines.append("  </tbody>")
    lines.append("</table>")

    return "\n".join(lines)


def normalize_html(html: str) -> str:
    """
    HTML을 정규화하여 비교 가능한 형태로 만든다.
    - 공백 정규화
    - 속성 순서 통일
    - 빈 속성 제거
    """
    html = re.sub(r"\s+", " ", html.strip())
    html = re.sub(r">\s+<", "><", html)

    # rowspan/colspan 기본값(=1) 제거
    html = re.sub(
        r"\s+colspan\s*=\s*(?:\"1\"|'1'|1)\b",
        "",
        html,
        flags=re.IGNORECASE,
    )
    html = re.sub(
        r"\s+rowspan\s*=\s*(?:\"1\"|'1'|1)\b",
        "",
        html,
        flags=re.IGNORECASE,
    )

    return html


def canonicalize_table_html(html: str) -> str:
    """테이블 HTML을 비교/후처리에 유리한 안정 포맷으로 정규화한다.

    - nested table이 없는 경우: parse_html_table -> structure_to_html 재직렬화
    - nested table이 있는 경우: 구조 재생성 시 중첩 정보가 손실될 수 있어
      lightweight normalize_html만 적용
    """
    fragment = extract_first_balanced_table(html) or str(html or "").strip()
    if not fragment:
        return ""

    # nested table은 lightweight 정규화만 수행
    if fragment.lower().count("<table") > 1:
        return normalize_html(fragment)

    try:
        structure = parse_html_table(fragment)
    except Exception:
        return normalize_html(fragment)

    return structure_to_html(structure, include_content=True)


def _reflow_cells_after_rowspan_adjust(
    cells: list[CellInfo],
    rows: int,
    cols: int,
) -> tuple[list[CellInfo], int, bool]:
    """rowspan 조정 이후 row-wise 순서를 기준으로 col을 재배치한다."""
    if rows <= 0 or cols <= 0:
        return cells, 0, False

    row_cells: dict[int, list[CellInfo]] = {}
    for cell in sorted(cells, key=lambda c: (c.row, c.col)):
        row_cells.setdefault(cell.row, []).append(cell)

    occupancy = [[False for _ in range(cols)] for _ in range(rows)]
    reflowed_cells: list[CellInfo] = []
    moved_col_count = 0

    for r in range(rows):
        cursor = 0
        for cell in row_cells.get(r, []):
            while cursor < cols and occupancy[r][cursor]:
                cursor += 1
            if cursor >= cols:
                return cells, 0, False

            new_col = cursor
            if new_col != cell.col:
                moved_col_count += 1

            new_cell = CellInfo(
                row=cell.row,
                col=new_col,
                rowspan=cell.rowspan,
                colspan=cell.colspan,
                content=cell.content,
                is_header=cell.is_header,
            )
            reflowed_cells.append(new_cell)

            for rr in range(new_cell.row, min(rows, new_cell.row + new_cell.rowspan)):
                for cc in range(new_col, min(cols, new_col + new_cell.colspan)):
                    occupancy[rr][cc] = True

            cursor = new_col + new_cell.colspan

    return reflowed_cells, moved_col_count, True


def postprocess_table_html(
    html: str,
    enable_repair: bool = True,
    fill_holes: bool = True,
) -> tuple[str, dict]:
    """모델 출력 테이블 HTML을 정규화/보정한다.

    Returns:
        (processed_html, info)
        info keys:
          - applied: 보정 결과 원문 대비 변경 여부
          - canonicalized: 정규화 적용 여부
          - repairs: 적용된 보정 카운트
          - issues: 보정 중 발견 이슈 목록
    """
    info = {
        "applied": False,
        "canonicalized": False,
        "repairs": 0,
        "issues": [],
    }

    def _add_issue(tag: str) -> None:
        if tag not in info["issues"]:
            info["issues"].append(tag)

    base = extract_first_balanced_table(html) or str(html or "").strip()
    if not base:
        return "", info
    base_norm = normalize_html(base)

    canonical = canonicalize_table_html(base)
    if not canonical:
        return "", info
    canonical_norm = normalize_html(canonical)
    canonical_changed = (canonical_norm != base_norm)

    if not enable_repair:
        if canonical_changed:
            info["canonicalized"] = True
            info["applied"] = True
        return canonical, info

    # nested table은 안전을 위해 구조 보정을 건너뛴다.
    if canonical.lower().count("<table") > 1:
        _add_issue("nested_table_skip_repair")
        return base, info

    try:
        structure = parse_html_table(canonical)
    except Exception:
        _add_issue("parse_error_after_canonical")
        return base, info

    rows = int(structure.num_rows)
    cols = int(structure.num_cols)
    if rows <= 0 or cols <= 0:
        _add_issue("empty_table")
        return base, info

    # parse_html_table의 num_cols는 입력 span outlier 영향을 받는다.
    # row별 colspan 합의 최빈값(동률 시 최소값)을 실효 열 수로 사용해 clamp 기준을 강화한다.
    #
    # 주의: rowspan이 있는 표에서는 "행별 colspan 합"이 실제 열 수를 과소추정하기 쉽다.
    # 이 경우 열 수 추론을 생략해 과보정으로 인한 대량 셀 삭제를 방지한다.
    has_rowspan_cell = any(int(cell.rowspan) > 1 for cell in structure.cells)
    row_width_by_row: dict[int, int] = {}
    for cell in structure.cells:
        row_width_by_row[cell.row] = row_width_by_row.get(cell.row, 0) + max(
            1, int(cell.colspan)
        )
    row_width_values = [w for w in row_width_by_row.values() if w > 0]
    target_cols = cols
    if row_width_values and not has_rowspan_cell:
        width_counter = Counter(row_width_values)
        max_count = max(width_counter.values())
        candidate_widths = [
            width for width, count in width_counter.items()
            if count == max_count
        ]
        inferred_cols = min(candidate_widths)
        if 0 < inferred_cols < target_cols:
            target_cols = inferred_cols
            info["repairs"] += 1
            _add_issue("cols_inferred_from_row_width_mode")

    repaired_cells: list[CellInfo] = []
    dropped_out_of_bounds = False
    rowspan_adjusted = False
    for cell in sorted(structure.cells, key=lambda c: (c.row, c.col)):
        # 범위 밖 셀은 제거
        if (
            cell.row < 0
            or cell.col < 0
            or cell.row >= rows
            or cell.col >= target_cols
        ):
            dropped_out_of_bounds = True
            info["repairs"] += 1
            _add_issue("dropped_out_of_bounds_cell")
            continue

        max_rowspan = max(1, rows - cell.row)
        max_colspan = max(1, target_cols - cell.col)
        # rowspan은 과보정 리스크가 높아 값 자체는 유지한다.
        # (명시 행 수를 넘어가는 overflow는 렌더러가 안전하게 처리할 수 있어
        # 단순 clamp로 인한 TEDS 하락을 피하는 편이 안정적이다.)
        new_rowspan = max(1, int(cell.rowspan))
        new_colspan = min(max(1, int(cell.colspan)), max_colspan)
        if int(cell.rowspan) > max_rowspan:
            _add_issue("rowspan_overflow_kept")
        if new_rowspan != cell.rowspan:
            rowspan_adjusted = True
        if new_rowspan != cell.rowspan or new_colspan != cell.colspan:
            info["repairs"] += 1
            _add_issue("span_clamped")

        repaired_cells.append(
            CellInfo(
                row=cell.row,
                col=cell.col,
                rowspan=new_rowspan,
                colspan=new_colspan,
                content=cell.content,
                is_header=cell.is_header,
            )
        )

    # 셀 삭제가 발생한 repair는 고위험 과보정일 수 있어 구조 보정을 롤백한다.
    if dropped_out_of_bounds:
        info["repairs"] = 0
        info["issues"] = [
            tag for tag in info["issues"]
            if tag not in {
                "cols_inferred_from_row_width_mode",
                "dropped_out_of_bounds_cell",
                "span_clamped",
            }
        ]
        _add_issue("repair_rolled_back_due_to_drop")
        return base, info

    if rowspan_adjusted:
        reflowed_cells, moved_col_count, ok = _reflow_cells_after_rowspan_adjust(
            repaired_cells,
            rows=rows,
            cols=target_cols,
        )
        if ok and moved_col_count > 0:
            repaired_cells = reflowed_cells
            info["repairs"] += 1
            _add_issue("rowspan_reflow")
        elif not ok:
            _add_issue("rowspan_reflow_skipped_no_slot")

    # occupancy 구성 후 hole 보정
    occupancy = [[-1 for _ in range(target_cols)] for _ in range(rows)]
    for idx, cell in enumerate(repaired_cells):
        for r in range(cell.row, min(rows, cell.row + cell.rowspan)):
            for c in range(cell.col, min(target_cols, cell.col + cell.colspan)):
                occupancy[r][c] = idx

    if fill_holes:
        holes_filled = 0
        for r in range(rows):
            for c in range(target_cols):
                if occupancy[r][c] != -1:
                    continue
                holes_filled += 1
                repaired_cells.append(
                    CellInfo(
                        row=r,
                        col=c,
                        rowspan=1,
                        colspan=1,
                        content="",
                        is_header=(r < structure.header_rows),
                    )
                )
        if holes_filled > 0:
            info["repairs"] += holes_filled
            _add_issue(f"holes_filled:{holes_filled}")

    if info["repairs"] <= 0:
        return base, info

    repaired_structure = TableStructure(
        num_rows=rows,
        num_cols=target_cols,
        cells=repaired_cells,
        header_rows=structure.header_rows,
        nested_table_count=0,
    )
    repaired_html = structure_to_html(repaired_structure, include_content=True)
    if normalize_html(repaired_html) != base_norm:
        info["applied"] = True
    else:
        # 보정 결과가 실질적으로 동일하면 원문 유지
        info["repairs"] = 0
        info["issues"] = []
        return base, info
    return repaired_html, info


def extract_spans_from_html(html: str) -> list[dict]:
    """
    HTML에서 span 정보만 추출한다.
    generate_thinking()에 전달할 형태로 반환.

    Returns:
        [{"row_start", "row_end", "col_start", "col_end", "content"}, ...]
    """
    structure = parse_html_table(html)
    spans = []
    for cell in structure.span_cells:
        spans.append({
            "row_start": cell.row,
            "row_end": cell.row_end,
            "col_start": cell.col,
            "col_end": cell.col_end,
            "content": cell.content,
        })
    return spans


def get_table_dimensions(html: str) -> tuple[int, int]:
    """HTML 테이블의 (행, 열) 크기를 반환."""
    structure = parse_html_table(html)
    return structure.num_rows, structure.num_cols


def html_to_structure_only(html: str) -> str:
    """셀 내용을 제거하고 구조만 남긴 HTML 반환 (TEDS-Structure용)."""
    structure = parse_html_table(html)
    return structure_to_html(structure, include_content=False)
