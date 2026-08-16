# synth/config.py
from dataclasses import dataclass
from typing import Tuple, List, Optional
import random


@dataclass
class SynthConfig:
    # reproducibility
    seed: int = 42

    # canvas
    img_size: Tuple[int, int] = (768, 768)
    margin: int = 32

    # table size
    rows_range: Tuple[int, int] = (5, 15)
    cols_range: Tuple[int, int] = (4, 12)

    # merge behavior (graph 단계에서 적용)
    enable_random_merge: bool = True
    merge_ratio: float = 0.30
    max_rowspan: int = 4
    max_colspan: int = 5

    # merge pattern weights
    merge_w_header: float = 0.35
    merge_w_leftcol: float = 0.30
    merge_w_random: float = 0.35

    # text + OCR-like
    empty_cell_ratio: float = 0.10
    multi_line_ratio: float = 0.25       # (render.py에서 스타일 텍스트 생성이 주로 담당)
    split_bbox_ratio: float = 0.35       # 셀 텍스트 bbox를 2~3개로 쪼개 many-to-one 유도

    # OCR bbox noise only (셀 경계는 정확히 맞추기)
    ocr_jitter_px: float = 2.0

    # line styles
    outer_thickness: int = 2
    inner_thickness: int = 1
    header_sep_thickness: int = 2

    # fonts
    font_paths: Optional[List[str]] = None

    # style preset
    # "policy" | "matrix" | "form" | "leave"
    table_style: str = "policy"

    def rng(self) -> random.Random:
        return random.Random(self.seed)
