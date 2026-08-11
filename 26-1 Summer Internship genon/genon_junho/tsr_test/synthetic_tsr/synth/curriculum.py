# synth/curriculum.py
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class CurriculumStage:
    name: str
    rows_range: Tuple[int,int]
    cols_range: Tuple[int,int]
    merge_ratio: float
    max_rowspan: int
    max_colspan: int
    empty_cell_ratio: float

def default_curriculum() -> List[CurriculumStage]:
    # 난이도: (1) no-merge → (2) light merge → (3) heavy merge → (4) stress
    return [
        CurriculumStage("easy_no_merge", (4,8), (3,8), 0.00, 1, 1, 0.05),
        CurriculumStage("mid_light_merge", (5,12), (4,10), 0.15, 2, 3, 0.08),
        CurriculumStage("mid_light_merge_2", (6,30), (5,12), 0.10, 8, 3, 0.12),
        CurriculumStage("hard_merge", (6,15), (5,12), 0.35, 4, 5, 0.12),
        # CurriculumStage("stress_complex", (10,18), (8,16), 0.55, 6, 6, 0.18),
    ]
