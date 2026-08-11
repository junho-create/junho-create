# synth/export_pubtabnet.py
from typing import Dict, Any
from .table_graph import TableGraph

def to_html(tg:TableGraph) -> str:
    # HTML: 각 셀의 rowspan/colspan 반영
    # occ grid를 순회하며 base cell일 때만 <td> 출력, 내부는 skip
    html = ["<table><tbody>"]
    for r in range(tg.rows):
        html.append("<tr>")
        c = 0
        while c < tg.cols:
            base_id = tg.occ[r][c]
            # base cell인지 확인
            cell = next((x for x in tg.cells if x.id == base_id), None)
            if cell is None:
                c += 1
                continue
            if not (cell.r == r and cell.c == c):
                c += 1
                continue

            attrs = []
            if cell.rowspan > 1:
                attrs.append(f'rowspan="{cell.rowspan}"')
            if cell.colspan > 1:
                attrs.append(f'colspan="{cell.colspan}"')
            attr_str = (" " + " ".join(attrs)) if attrs else ""
            html.append(f"<td{attr_str}></td>")
            c += cell.colspan
        html.append("</tr>")
    html.append("</tbody></table>")
    return "".join(html)

def export_pubtabnet_record(image_filename:str, tg:TableGraph) -> Dict[str, Any]:
    return {
        "filename": image_filename,
        "html": {"structure": {"tokens": [to_html(tg)]}},  # 학습 코드에 맞춰 토큰 분해 가능
        "split": "train"
    }
