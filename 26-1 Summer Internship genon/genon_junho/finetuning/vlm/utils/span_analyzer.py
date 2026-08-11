"""
Span 복잡도 분석기
- 데이터셋의 span 분포 분석
- 복잡도 기반 데이터 분류
- 학습 데이터 밸런싱
"""

from collections import Counter
from dataclasses import dataclass
from typing import Optional

from utils.html_utils import parse_html_table, TableStructure


@dataclass
class SpanStats:
    """개별 테이블의 span 통계."""
    total_cells: int = 0
    span_cells: int = 0
    colspan_cells: int = 0
    rowspan_cells: int = 0
    both_span_cells: int = 0      # colspan + rowspan 동시
    max_colspan: int = 1
    max_rowspan: int = 1
    complexity: str = "simple"     # simple / medium / complex (하위 호환)
    complexity_score: float = 0.0  # 0.0 ~ 1.0
    # 세분화 필드
    span_pattern: str = "none"           # col_only / row_only / mixed / none
    max_span: int = 1                    # max(max_colspan, max_rowspan)
    grid_irregularity: float = 0.0      # 0.0 ~ 1.0
    table_size_cells: int = 0           # num_rows × num_cols
    nested_table_count: int = 0         # 중첩 table 태그 수

    @property
    def span_ratio(self) -> float:
        if self.total_cells == 0:
            return 0.0
        return self.span_cells / self.total_cells


def table_size_category(size_cells: int) -> str:
    """그리드 크기(num_rows × num_cols)를 범주로 분류한다."""
    if size_cells <= 20:
        return "small"
    if size_cells <= 80:
        return "medium"
    return "large"


def max_span_category(max_span: int) -> str:
    """최대 span 크기를 범주로 분류한다."""
    if max_span <= 3:
        return "normal"
    if max_span <= 6:
        return "large"
    return "very_large"


def grid_irregularity_category(irregularity: float) -> str:
    """그리드 불규칙도를 범주로 분류한다."""
    if irregularity <= 0.10:
        return "regular"
    if irregularity <= 0.30:
        return "moderate"
    return "irregular"


@dataclass
class DatasetSpanProfile:
    """데이터셋 전체의 span 분포."""
    total_samples: int = 0
    simple_count: int = 0          # span 없음
    medium_count: int = 0          # span 1~2개
    complex_count: int = 0         # span 3개 이상
    avg_span_ratio: float = 0.0
    colspan_distribution: dict = None   # {span_size: count}
    rowspan_distribution: dict = None

    def __post_init__(self):
        if self.colspan_distribution is None:
            self.colspan_distribution = {}
        if self.rowspan_distribution is None:
            self.rowspan_distribution = {}

    @property
    def composition(self) -> dict[str, float]:
        """복잡도별 비율."""
        total = self.total_samples or 1
        return {
            "simple": self.simple_count / total,
            "medium": self.medium_count / total,
            "complex": self.complex_count / total,
        }

    def summary(self) -> str:
        comp = self.composition
        return (
            f"Dataset Span Profile ({self.total_samples} samples)\n"
            f"  Simple (no span):     {self.simple_count:>5} ({comp['simple']:.1%})\n"
            f"  Medium (1-2 spans):   {self.medium_count:>5} ({comp['medium']:.1%})\n"
            f"  Complex (3+ spans):   {self.complex_count:>5} ({comp['complex']:.1%})\n"
            f"  Avg span ratio:       {self.avg_span_ratio:.3f}\n"
            f"  Colspan dist:         {dict(sorted(self.colspan_distribution.items()))}\n"
            f"  Rowspan dist:         {dict(sorted(self.rowspan_distribution.items()))}"
        )


def analyze_span(html: str) -> SpanStats:
    """
    단일 테이블 HTML의 span 통계를 분석한다.

    Args:
        html: <table>...</table> HTML

    Returns:
        SpanStats 객체
    """
    structure = parse_html_table(html)
    stats = SpanStats()

    stats.total_cells = len(structure.cells)
    stats.colspan_cells = len(structure.colspan_cells)
    stats.rowspan_cells = len(structure.rowspan_cells)
    stats.both_span_cells = len([
        c for c in structure.cells if c.colspan > 1 and c.rowspan > 1
    ])
    stats.span_cells = len(structure.span_cells)

    if structure.span_cells:
        stats.max_colspan = max(c.colspan for c in structure.cells)
        stats.max_rowspan = max(c.rowspan for c in structure.cells)

    # 복잡도 계산 (하위 호환: complex_* → complex)
    stats.complexity = structure.complexity_category
    stats.complexity_score = _compute_complexity_score(structure)

    # 세분화 필드 계산
    stats.span_pattern = structure.span_pattern
    stats.max_span = max(stats.max_colspan, stats.max_rowspan)
    stats.grid_irregularity = round(structure.grid_irregularity, 4)
    stats.table_size_cells = structure.num_rows * structure.num_cols
    stats.nested_table_count = structure.nested_table_count

    return stats


def _compute_complexity_score(structure: TableStructure) -> float:
    """
    테이블의 span 복잡도 점수를 0.0~1.0으로 계산한다.

    고려 요소:
    - span 셀 개수
    - span 크기 (colspan*rowspan)
    - 전체 셀 대비 span 비율
    - colspan+rowspan 동시 존재 여부
    """
    if not structure.span_cells:
        return 0.0

    n_cells = len(structure.cells) or 1
    n_spans = len(structure.span_cells)

    # Factor 1: span 비율 (0~0.3)
    span_ratio = min(n_spans / n_cells, 1.0) * 0.3

    # Factor 2: span 크기 (0~0.3)
    max_span_area = max(c.colspan * c.rowspan for c in structure.span_cells)
    span_size_score = min(max_span_area / 10.0, 1.0) * 0.3

    # Factor 3: span 다양성 (0~0.2)
    has_colspan = any(c.colspan > 1 for c in structure.span_cells)
    has_rowspan = any(c.rowspan > 1 for c in structure.span_cells)
    has_both = any(c.colspan > 1 and c.rowspan > 1 for c in structure.span_cells)
    diversity_score = (int(has_colspan) + int(has_rowspan) + int(has_both)) / 3.0 * 0.2

    # Factor 4: span 개수 (0~0.2)
    count_score = min(n_spans / 5.0, 1.0) * 0.2

    return span_ratio + span_size_score + diversity_score + count_score


def analyze_dataset(html_list: list[str]) -> DatasetSpanProfile:
    """
    데이터셋 전체의 span 분포를 분석한다.

    Args:
        html_list: HTML 문자열 리스트

    Returns:
        DatasetSpanProfile 객체
    """
    profile = DatasetSpanProfile()
    profile.total_samples = len(html_list)

    colspan_counter = Counter()
    rowspan_counter = Counter()
    total_span_ratio = 0.0

    for html in html_list:
        try:
            stats = analyze_span(html)
        except Exception:
            continue

        # 복잡도 분류
        if stats.complexity == "simple":
            profile.simple_count += 1
        elif stats.complexity == "medium":
            profile.medium_count += 1
        else:
            profile.complex_count += 1

        total_span_ratio += stats.span_ratio

        # span 크기 분포 집계
        structure = parse_html_table(html)
        for cell in structure.span_cells:
            if cell.colspan > 1:
                colspan_counter[cell.colspan] += 1
            if cell.rowspan > 1:
                rowspan_counter[cell.rowspan] += 1

    profile.avg_span_ratio = total_span_ratio / max(len(html_list), 1)
    profile.colspan_distribution = dict(colspan_counter)
    profile.rowspan_distribution = dict(rowspan_counter)

    return profile


def categorize_by_complexity(
    html_list: list[str],
) -> dict[str, list[int]]:
    """
    HTML 리스트를 복잡도별로 인덱스 분류한다.

    Returns:
        {"simple": [idx, ...], "medium": [idx, ...], "complex": [idx, ...]}
    """
    categories = {"simple": [], "medium": [], "complex": []}
    for idx, html in enumerate(html_list):
        try:
            stats = analyze_span(html)
            categories[stats.complexity].append(idx)
        except Exception:
            categories["simple"].append(idx)  # 파싱 실패 시 simple로 분류
    return categories


def compute_sampling_weights(
    html_list: list[str],
    target_ratios: Optional[dict[str, float]] = None,
) -> list[float]:
    """
    학습 시 span 데이터에 가중치를 부여하기 위한 샘플링 가중치 계산.

    Args:
        html_list: HTML 리스트
        target_ratios: 목표 비율. 기본값: {"simple": 0.2, "medium": 0.3, "complex": 0.5}

    Returns:
        각 샘플의 가중치 리스트 (합계 = len(html_list))
    """
    if target_ratios is None:
        target_ratios = {"simple": 0.2, "medium": 0.3, "complex": 0.5}

    categories = categorize_by_complexity(html_list)
    n = len(html_list)

    weights = [1.0] * n
    for cat, target_ratio in target_ratios.items():
        indices = categories[cat]
        if not indices:
            continue
        current_ratio = len(indices) / n
        if current_ratio > 0:
            w = target_ratio / current_ratio
            for idx in indices:
                weights[idx] = w

    # 정규화: 합계 = n
    total = sum(weights)
    weights = [w * n / total for w in weights]

    return weights
