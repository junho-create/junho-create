# synth/table_graph.py
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random

@dataclass
class Cell:
    r: int
    c: int
    rowspan: int = 1
    colspan: int = 1
    text: str = ""
    is_empty: bool = False
    id: int = -1  # logical cell id (unique)

@dataclass
class TableGraph:
    rows: int
    cols: int
    cells: List[Cell]
    occ: List[List[Optional[int]]]  # occupancy grid -> cell.id

def _init_grid(rows:int, cols:int) -> TableGraph:
    cells = []
    occ = [[None for _ in range(cols)] for _ in range(rows)]
    cid = 0
    for r in range(rows):
        for c in range(cols):
            cell = Cell(r=r, c=c, id=cid)
            cells.append(cell)
            occ[r][c] = cid
            cid += 1
    return TableGraph(rows=rows, cols=cols, cells=cells, occ=occ)

def _can_merge(tg:TableGraph, r:int, c:int, rs:int, cs:int) -> bool:
    if r+rs > tg.rows or c+cs > tg.cols:
        return False
    # 영역이 모두 단일셀(merge 전)이고 동일 시작점 포함인지 확인
    for rr in range(r, r+rs):
        for cc in range(c, c+cs):
            if tg.occ[rr][cc] is None:
                return False
            # 이미 다른 merge에 포함된 셀이면 거부
            cid = tg.occ[rr][cc]
            cell = tg.cells[cid]
            if not (cell.r == rr and cell.c == cc and cell.rowspan == 1 and cell.colspan == 1):
                return False
    return True

def _apply_merge(tg:TableGraph, r:int, c:int, rs:int, cs:int):
    base_id = tg.occ[r][c]
    base = tg.cells[base_id]
    base.rowspan = rs
    base.colspan = cs
    # 나머지 칸은 base에 점유되도록 표시하고, 해당 셀은 "비활성" 처리
    for rr in range(r, r+rs):
        for cc in range(c, c+cs):
            tg.occ[rr][cc] = base_id
            if rr == r and cc == c:
                continue
            other = tg.cells[rr*tg.cols + cc]
            other.id = -1  # invalidate logical cell (not emitted)
    return tg

# synth/table_graph.py (교체)
def build_table_graph(
    rows:int, cols:int, rng:random.Random,
    merge_ratio:float, max_rowspan:int, max_colspan:int,
    enable_random_merge: bool = True,
    w_header: float = 0.35,
    w_leftcol: float = 0.30,
    w_random: float = 0.35,
) -> TableGraph:
    tg = _init_grid(rows, cols)

    if not enable_random_merge or merge_ratio <= 0:
        # 활성 셀만 남기기(기본 그리드)
        tg.cells = [c for c in tg.cells if c.id != -1]
        for i, c in enumerate(tg.cells):
            c.id = i
        tg.occ = [[r*cols+c for c in range(cols)] for r in range(rows)]
        return tg

    # 목표 merge 개수(대략)
    target_merges = max(1, int(rows * cols * merge_ratio))
    max_attempts = target_merges * 20  # 실패 대비 여유

    weights = [w_header, w_leftcol, w_random]
    wsum = sum(weights) if sum(weights) > 0 else 1.0

    def pick_anchor():
        p = rng.random() * wsum
        if p < w_header:
            # header row에서 colspan 위주
            r = 0
            c = rng.randrange(0, max(1, cols-1))
            return r, c, "header"
        p -= w_header
        if p < w_leftcol:
            # left column에서 rowspan 위주
            r = rng.randrange(0, max(1, rows-1))
            c = 0
            return r, c, "leftcol"
        # random
        r = rng.randrange(0, rows)
        c = rng.randrange(0, cols)
        return r, c, "random"

    merged = 0
    attempts = 0

    while merged < target_merges and attempts < max_attempts:
        attempts += 1
        r, c, mode = pick_anchor()

        # span 샘플링: mode에 따라 편향
        if mode == "header":
            rs = 1
            cs = rng.randint(2, min(max_colspan, cols - c))
        elif mode == "leftcol":
            rs = rng.randint(2, min(max_rowspan, rows - r))
            cs = 1
        else:
            # rs = rng.randint(1, min(max_rowspan, rows - r))
            # cs = rng.randint(1, min(max_colspan, cols - c))

            # random mode에서도 rowspan / colspan 중 하나만 허용
            merge_dir = rng.choice(["row", "col"])
            if merge_dir == "row":
                rs = rng.randint(2, min(max_rowspan, rows - r)) if rows - r >= 2 else 1
                cs = 1
            else:
                rs = 1
                cs = rng.randint(2, min(max_colspan, cols - c)) if cols - c >= 2 else 1

        if rs == 1 and cs == 1:
            continue

        # 겹침/이미-merge 포함 여부 검사
        if _can_merge(tg, r, c, rs, cs):
            _apply_merge(tg, r, c, rs, cs)
            merged += 1

    # 활성 셀만 남기기
    tg.cells = [c for c in tg.cells if c.id != -1]
    # id 재할당(연속)
    for i, c in enumerate(tg.cells):
        c.id = i

    # occ 재구성
    new_occ = [[None for _ in range(cols)] for _ in range(rows)]
    for rr in range(rows):
        for cc in range(cols):
            base_id = None
            for cell in tg.cells:
                if cell.r <= rr < cell.r+cell.rowspan and cell.c <= cc < cell.c+cell.colspan:
                    base_id = cell.id
                    break
            new_occ[rr][cc] = base_id
    tg.occ = new_occ
    return tg
