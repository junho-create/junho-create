# synth/export_tflop.py
from typing import Dict, Any, List, Tuple
from .table_graph import TableGraph, Cell


def _row_major_reindex(tg: TableGraph) -> Tuple[TableGraph, Dict[int,int]]:
    """cell.id를 (r,c) 오름차순으로 재할당하고 occ도 그에 맞게 갱신."""
    old_cells = list(tg.cells)

    ordered = sorted(old_cells, key=lambda x: (x.r, x.c))
    old_to_new = {c.id: i for i, c in enumerate(ordered)}

    for i, c in enumerate(ordered):
        c.id = i

    new_occ = [[None for _ in range(tg.cols)] for _ in range(tg.rows)]
    for r in range(tg.rows):
        for c in range(tg.cols):
            oid = tg.occ[r][c]
            new_occ[r][c] = old_to_new[oid] if oid is not None else None

    tg.cells = ordered
    tg.occ = new_occ
    return tg, old_to_new


def _grid_tags(tg: TableGraph) -> List[List[str]]:
    tags = [["" for _ in range(tg.cols)] for _ in range(tg.rows)]
    id_to_cell = {c.id: c for c in tg.cells}

    for r in range(tg.rows):
        for c in range(tg.cols):
            cid = tg.occ[r][c]
            cell = id_to_cell[cid]
            if cell.r == r and cell.c == c:
                tags[r][c] = "C-tag"
            else:
                is_left = (r == cell.r and c > cell.c)
                is_up   = (c == cell.c and r > cell.r)
                if (r > cell.r) and (c > cell.c):
                    tags[r][c] = "X-tag"
                elif is_left:
                    tags[r][c] = "L-tag"
                elif is_up:
                    tags[r][c] = "U-tag"
                else:
                    tags[r][c] = "X-tag"
    return tags


def build_otsl_seq_like_tflop(tg: TableGraph, header_rows: int = 1) -> List[str]:
    grid = _grid_tags(tg)
    seq: List[str] = []

    seq.append("<thead>")
    for r in range(min(header_rows, tg.rows)):
        for c in range(tg.cols):
            seq.append(grid[r][c])
        seq.append("NL-tag")
    seq.append("</thead>")

    seq.append("<tbody>")
    for r in range(header_rows, tg.rows):
        for c in range(tg.cols):
            seq.append(grid[r][c])
        seq.append("NL-tag")
    seq.append("</tbody>")
    return seq


def build_org_html_tokens_like_dataset(tg: TableGraph, header_rows: int = 1) -> List[str]:
    id_to_cell = {c.id: c for c in tg.cells}

    def emit_section(rows_range: range) -> List[str]:
        out: List[str] = []
        for r in rows_range:
            out.append("<tr>")
            c = 0
            while c < tg.cols:
                cid = tg.occ[r][c]
                cell = id_to_cell[cid]
                if not (cell.r == r and cell.c == c):
                    c += 1
                    continue

                out.append("<td")
                if cell.rowspan > 1:
                    out.append(f' rowspan="{cell.rowspan}"')
                if cell.colspan > 1:
                    out.append(f' colspan="{cell.colspan}"')
                out.append(">")
                out.append("</td>")
                c += cell.colspan
            out.append("</tr>")
        return out

    html: List[str] = []
    html.append("<thead>")
    html += emit_section(range(0, min(header_rows, tg.rows)))
    html.append("</thead>")
    html.append("<tbody>")
    html += emit_section(range(header_rows, tg.rows))
    html.append("</tbody>")
    return html


def export_tflop_record_like_dataset(
    image_filename: str,
    tg: TableGraph,
    cell_polys: Dict[int, List[Tuple[float,float]]],
    ocr_boxes: List[Dict[str,Any]],
    split: str = "train",
    header_rows: int = 1,
) -> Dict[str, Any]:
    """
    변경사항 요약:
    - gold_coord: 이전 요구사항 유지
    - dr_coord: 셀 단위로 묶음
        <cell_id>: [
            [모든 OCR bbox],
            <gold_coord index (=cell_id)>,
            <셀에 속한 모든 텍스트를 합친 문자열>
        ]
    """

    # 1) cell id row-major 재정렬
    tg, old_to_new = _row_major_reindex(tg)

    # 2) cell_polys 재매핑
    cell_polys = {
        old_to_new[oid]: poly
        for oid, poly in cell_polys.items()
        if oid in old_to_new
    }

    # 3) ocr_boxes 재매핑
    new_ocr_boxes = []
    for b in ocr_boxes:
        oid = int(b["cell_id"])
        if oid in old_to_new:
            bb = dict(b)
            bb["cell_id"] = old_to_new[oid]
            new_ocr_boxes.append(bb)
    ocr_boxes = new_ocr_boxes

    # cell_id → OCR boxes 그룹핑
    cell_to_boxes: Dict[int, List[Dict[str, Any]]] = {}
    for b in ocr_boxes:
        cid = int(b["cell_id"])
        cell_to_boxes.setdefault(cid, []).append(b)

    # 4) gold_coord 생성 (기존 규칙 유지)
    gold_coord: List[str] = []
    has_ocr_by_cell = set(int(b["cell_id"]) for b in ocr_boxes)

    for cell in tg.cells:
        if cell.id in has_ocr_by_cell:
            poly = cell_polys[cell.id]
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

            boxes = []
            texts = []
            for b in cell_to_boxes.get(cell.id, []):
                boxes.append([float(v) for v in b["bbox"]])
                txt = str(b.get("text", "")).strip()
                if txt:
                    texts.append(txt)

            merged_text = " ".join(texts)

            gold_coord.append(f"{x1} {y1} {x2} {y2} 2 {merged_text}")
        else:
            gold_coord.append("-1 -1 -1 -1 1")

    # --------------------------------------------------
    # 5) dr_coord 생성 (⭐ 핵심 수정 부분)
    # --------------------------------------------------
    dr_coord: Dict[str, Any] = {}

    dr_id = 0
    for cell in tg.cells:
        cid = cell.id
        boxes = []
        texts = []

        for b in cell_to_boxes.get(cid, []):
            boxes.append([float(v) for v in b["bbox"]])
            txt = str(b.get("text", "")).strip()
            if txt:
                texts.append(txt)

        merged_text = " ".join(texts)
        if len(merged_text) == 0:
            continue

        dr_coord[str(dr_id)] = [
            boxes,        # 동일 셀의 모든 OCR bbox
            cid,          # gold_coord index
            merged_text,  # 셀 전체 텍스트
        ]
        dr_id += 1

    # 6) otsl / html (기존 유지)
    otsl_seq = build_otsl_seq_like_tflop(tg, header_rows=header_rows)
    org_html = build_org_html_tokens_like_dataset(tg, header_rows=header_rows)

    return {
        "file_name": image_filename,
        "dr_coord": dr_coord,
        "gold_coord": gold_coord,
        "org_html": org_html,
        "otsl_seq": otsl_seq,
        "num_rows": tg.rows,
        "num_cols": tg.cols,
        "split": split,
    }
