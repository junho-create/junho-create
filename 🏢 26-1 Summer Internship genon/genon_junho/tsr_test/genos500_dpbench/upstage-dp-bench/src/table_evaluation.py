"""
Most of the code in this file is derived from the paper "Image-based table recognition: data, model, and evaluation".
The original paper can be accessed at: https://arxiv.org/pdf/1911.10683.
The code is available at: https://github.com/ibm-aur-nlp/PubTabNet.
A slight modification has been added to the code to improve the evaluation process.
"""

import re
import distance

from lxml import etree, html
from collections import deque
from apted.helpers import Tree
from apted import APTED, Config


class TableTree(Tree):
    """Table Tree class for APTED"""
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == 'td':
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % \
                     (self.tag, self.colspan, self.rowspan, self.content)
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    """Custom Configuration for APTED"""
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value"""
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1"""
        return float(distance.levenshtein(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees"""
        if (node1.tag != node2.tag) or \
                (node1.colspan != node2.colspan) or \
                (node1.rowspan != node2.rowspan):
            return 1.
        if node1.tag == 'td':
            if node1.content or node2.content:
                return self.normalized_distance(
                    node1.content, node2.content
                )
        return 0.


class TEDSEvaluator(object):
    """Tree Edit Distance basead Similarity"""
    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        assert isinstance(n_jobs, int) and (n_jobs >= 1), (
            'n_jobs must be an integer greather than 1'
        )
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        """Tokenizes table cells"""
        self.__tokens__.append('<%s>' % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != 'unk':
            self.__tokens__.append('</%s>' % node.tag)
        if node.tag != 'td' and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        """Converts HTML tree to the format required by apted"""
        global __tokens__
        if node.tag == 'td':
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(
                node.tag,
                int(node.attrib.get('colspan', '1')),
                int(node.attrib.get('rowspan', '1')),
                cell, *deque()
            )
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != 'td':
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred, true):
        """Computes TEDS score between the prediction and the ground truth of a given sample"""
        if (not pred) or (not true):
            return 0.0
        parser = html.HTMLParser(remove_comments=True, encoding='utf-8')
        pred = html.fromstring(pred, parser=parser)
        true = html.fromstring(true, parser=parser)

        if pred.xpath('body/table') and true.xpath('body/table'):
            pred = pred.xpath('body/table')[0]
            true = true.xpath('body/table')[0]
            if self.ignore_nodes:
                etree.strip_tags(pred, *self.ignore_nodes)
                etree.strip_tags(true, *self.ignore_nodes)
            n_nodes_pred = len(pred.xpath('.//*'))
            n_nodes_true = len(true.xpath('.//*'))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            tree_pred = self.load_html_tree(pred)
            tree_true = self.load_html_tree(true)
            distance = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
            return 1.0 - (float(distance) / n_nodes)
        else:
            return 0.0


def get_table_contents(text):
    # Regular expression to capture content within <table ...> and </table> tags
    table_contents = re.findall(r'<table[^>]*?>(.*?)</table>', text, flags=re.DOTALL)

    if len(table_contents) == 0:
        table_contents = [text]

    return table_contents


def normalize_table_header_tags(table_html: str) -> str:
    if not table_html:
        return table_html

    # Normalize header tags for robust cross-parser comparison.
    normalized = re.sub(r'<\s*th\b', '<td', table_html, flags=re.IGNORECASE)
    normalized = re.sub(r'</\s*th\s*>', '</td>', normalized, flags=re.IGNORECASE)

    return normalized


# ---------------------------------------------------------------------------
# BBox 기반 테이블 매칭 (이슈 #272: https://github.com/genonai/doc_parser/issues/272)
#
# 현행(index) 방식: 문서 단위로 GT/Pred 의 첫 테이블만 비교한다. Pred 가 앞쪽
# 테이블을 놓치면 이후 테이블 순서가 밀려 GT 와 엉뚱하게 매칭된다.
# 여기서는 각 테이블의 bbox 를 IoU 로 매칭해 이 문제를 없앤다.
#
# [좌표계 정규화]
# 파이프라인마다 테이블 좌표의 기준이 다르므로, 모두 "실제 페이지의 [0,1],
# 원점 좌상단" 캐노니컬 프레임으로 변환한 뒤 IoU 를 잰다. 각 측이 자기
# 좌표공간(coord space)을 선언한다:
#   - extent      : 그 문서 전체 element 좌표의 최대범위로 [0,1] 정규화(원점0,
#                   상단기준). 이미지 파일 불필요. (기본값, 같은 좌표계끼리 빠른 점검용)
#   - image       : 실제 이미지 픽셀좌표 -> 이미지 크기로 나눔(원점 좌상단).
#                   --images_dir 필요. (GT, 또는 픽셀좌표를 내는 파이프라인)
#   - grid        : 0~N 정규화 그리드 -> N(기본 1024)으로 나눔(원점 좌상단).
#                   (Qwen3.5 파인튜닝 JSON 출력; 모델 bbox_scale=1024 와 일치)
#   - fraction_tl : 이미 [0,1] 비율, 원점 좌상단 -> 그대로.
#   - fraction_bl : 이미 [0,1] 비율, 원점 좌하단(PDF) -> Y 뒤집기.
#                   (Genos doc_parser v1.3.8 / v2.0 출력)
# 서로 다른 파이프라인을 공정 비교하려면 GT 와 Pred 를 같은 프레임으로 맞춰야
# 하므로, image/grid/fraction_* 를 쓸 때는 GT 도 image(--images_dir)로 둔다.
# ---------------------------------------------------------------------------

COORD_SPACES = ("extent", "image", "grid", "fraction_tl", "fraction_bl")


def _table_bbox(elem: dict):
    """elem['coordinates'](4점 폴리곤) -> [x0,y0,x1,y1]. 없으면 None."""
    coords = elem.get('coordinates')
    if not coords:
        return None
    xs = [p.get('x', 0) for p in coords]
    ys = [p.get('y', 0) for p in coords]
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def _page_extent(data: dict):
    """문서 내 모든 element 좌표의 최대 (x, y). extent 정규화 기준(원점 0)."""
    max_x, max_y = 0.0, 0.0
    for elem in data.get('elements', []):
        coords = elem.get('coordinates') or []
        for p in coords:
            max_x = max(max_x, p.get('x', 0))
            max_y = max(max_y, p.get('y', 0))
    return max_x, max_y


_IMG_INDEX_CACHE = {}  # images_dir -> {stem(소문자): 파일경로}
_IMG_DIMS_CACHE = {}   # (images_dir, stem) -> (w, h)
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _build_image_index(images_dir: str):
    """images_dir 를 (하위폴더 포함) 한 번 훑어 stem->경로 인덱스를 만든다."""
    import os
    index = {}
    for root, _dirs, files in os.walk(images_dir):
        for fn in files:
            stem, ext = os.path.splitext(fn)
            if ext.lower() in _IMG_EXTS:
                # 먼저 발견한 것을 우선(중복 stem 이면 유지)
                index.setdefault(stem.lower(), os.path.join(root, fn))
    return index


def _image_dims(images_dir: str, image_key: str):
    """image_key('doc_0001.pdf') 에 대응하는 실제 이미지의 (w, h). 없으면 None.

    images_dir 하위(중첩 포함)에서 파일명(확장자 제외)이 image_key 의 stem 과
    일치하는 이미지를 찾아 PIL 로 실제 크기를 읽는다.
    """
    import os
    stem = os.path.splitext(os.path.basename(image_key))[0].lower()
    cache_key = (images_dir, stem)
    if cache_key in _IMG_DIMS_CACHE:
        return _IMG_DIMS_CACHE[cache_key]

    if images_dir not in _IMG_INDEX_CACHE:
        _IMG_INDEX_CACHE[images_dir] = _build_image_index(images_dir)
    path = _IMG_INDEX_CACHE[images_dir].get(stem)

    dims = None
    if path:
        from PIL import Image  # image 좌표공간 사용 시에만 필요
        with Image.open(path) as im:
            dims = im.size  # (w, h)
    _IMG_DIMS_CACHE[cache_key] = dims
    return dims


def _normalize_to_canonical(bbox, space, *, page_extent=None, image_dims=None, grid=1024.0):
    """raw bbox[x0,y0,x1,y1] -> 캐노니컬 [0,1] 좌상단 프레임. 불가하면 None."""
    if bbox is None:
        return None
    if space == "extent":
        w = page_extent[0] if page_extent and page_extent[0] > 0 else 1.0
        h = page_extent[1] if page_extent and page_extent[1] > 0 else 1.0
        return [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]
    if space == "image":
        if not image_dims:
            return None
        w, h = image_dims
        return [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]
    if space == "grid":
        g = grid if grid > 0 else 1024.0
        return [bbox[0] / g, bbox[1] / g, bbox[2] / g, bbox[3] / g]
    if space == "fraction_tl":
        return [bbox[0], bbox[1], bbox[2], bbox[3]]
    if space == "fraction_bl":
        # 원점 좌하단 -> 좌상단: y' = 1 - y (상/하 경계 교환)
        return [bbox[0], 1.0 - bbox[3], bbox[2], 1.0 - bbox[1]]
    raise ValueError(f"unknown coord space: {space}")


def _table_content_sha1(html_text: str) -> str:
    """표 HTML의 내용 지문. 태그/공백/따옴표·th↔td 차이를 무시하고 셀 텍스트만
    이어붙여 sha1. 매칭 방식 간 '같은 pred 를 골랐나' 비교용 (issue #318)."""
    import hashlib
    import re as _re
    text = _re.sub(r"<[^>]+>", "|", html_text or "")
    text = _re.sub(r"[\s|]+", "", text)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _wrap_one_table(table_html: str, normalize_header_tags: bool) -> str:
    """단일 table element 의 html 을 dp-bench 채점용 문자열로 감싼다.

    extract_tables 와 동일 규칙(thead/tbody 제거는 calc_table_score 에서 수행).
    """
    parts = ''
    for inner in get_table_contents(table_html):
        if normalize_header_tags:
            inner = normalize_table_header_tags(inner)
        parts += f'<table>{inner}</table>'
    return f'<html><body>{parts}</body></html>'


def _doc_tables_with_bbox(
    data: dict,
    normalize_header_tags: bool,
    space: str = "extent",
    image_dims=None,
    grid: float = 1024.0,
):
    """문서에서 (wrapped_html, canonical_bbox) 리스트를 반환.

    canonical_bbox 는 [0,1] 좌상단 프레임. bbox 가 없거나 0 면적이면 None.
    """
    page_extent = _page_extent(data) if space == "extent" else None
    out = []
    for elem in data.get('elements', []):
        if elem.get('category', '').lower() != 'table':
            continue
        wrapped = _wrap_one_table(elem['content']['html'], normalize_header_tags)
        raw = _table_bbox(elem)
        nb = None
        if raw is not None and (raw[2] - raw[0]) > 0 and (raw[3] - raw[1]) > 0:
            nb = _normalize_to_canonical(
                raw, space, page_extent=page_extent,
                image_dims=image_dims, grid=grid,
            )
        out.append((wrapped, nb))
    return out


def _greedy_match(gt_boxes, pred_boxes, iou_thr: float):
    """GT->Pred 그리디 IoU 매칭. {gt_idx: pred_idx} 반환 (임계값 미만 제외)."""
    pairs = []
    for gi, gb in enumerate(gt_boxes):
        if gb is None:
            continue
        for pj, pb in enumerate(pred_boxes):
            if pb is None:
                continue
            iou = _iou(gb, pb)
            if iou >= iou_thr:
                pairs.append((iou, gi, pj))
    pairs.sort(reverse=True)
    matched, used_pred = {}, set()
    for _iou_val, gi, pj in pairs:
        if gi in matched or pj in used_pred:
            continue
        matched[gi] = pj
        used_pred.add(pj)
    return matched


def prepare_table_dataset_bbox(
    gt_data,
    pred_data,
    normalize_pred_table_header_tags: bool = True,
    iou_thr: float = 0.5,
    unmatched_gt: str = "zero",
    gt_coord_space: str = "extent",
    pred_coord_space: str = "extent",
    images_dir: str = None,
    pred_grid: float = 1024.0,
    match_dump_path: str = None,
):
    """bbox 매칭 기반으로 (GT, Pred) 테이블 문자열 쌍 리스트를 만든다.

    채점 단위가 '문서'가 아니라 '개별 테이블'이다.
      - 매칭된 GT 테이블: (gt_html, 매칭된 pred_html) 쌍
      - 매칭 안 된 GT 테이블(= pred 미검출):
          unmatched_gt='zero' -> (gt_html, '')  => TEDS 0 점 처리 (옵션 b)
          unmatched_gt='skip' -> 채점에서 제외 (옵션 a)

    gt_coord_space / pred_coord_space: 각 측 좌표공간(COORD_SPACES 참조).
    image 공간을 쓰면 images_dir 에서 페이지 크기를 읽는다.

    match_dump_path 를 주면 문서별 GT↔Pred 매칭 쌍(iou 포함)을 JSON 으로
    남긴다 — 매칭 방식 간 비교 분석용 (이슈 doc_parser#318 Phase 3).

    Returns:
        (gt_table_list, pred_table_list, stats)
    """
    if gt_coord_space == "image" or pred_coord_space == "image":
        if not images_dir:
            raise ValueError(
                "coord space 'image' requires --images_dir to read page sizes."
            )

    gt_table_list, pred_table_list = [], []
    stats = {
        "gt_tables": 0, "matched": 0, "unmatched_gt": 0,
        "extra_pred": 0, "degenerate_pred_docs": 0, "missing_image_docs": 0,
    }
    match_dump = {}

    for image_key in gt_data.keys():
        img_dims = None
        if gt_coord_space == "image" or pred_coord_space == "image":
            img_dims = _image_dims(images_dir, image_key)
            if img_dims is None:
                stats["missing_image_docs"] += 1

        gt_tables = _doc_tables_with_bbox(
            gt_data.get(image_key, {}), normalize_header_tags=False,
            space=gt_coord_space, image_dims=img_dims, grid=pred_grid)
        if not gt_tables:
            continue

        pred_elem = pred_data.get(image_key, {}) or {}
        pred_tables = _doc_tables_with_bbox(
            pred_elem, normalize_header_tags=normalize_pred_table_header_tags,
            space=pred_coord_space, image_dims=img_dims, grid=pred_grid)

        # pred 에 table 은 있는데 bbox 가 전부 비어있으면(HTML 출력 등) 매칭 불가
        if pred_tables and all(b is None for _, b in pred_tables):
            stats["degenerate_pred_docs"] += 1

        gt_boxes = [b for _, b in gt_tables]
        pred_boxes = [b for _, b in pred_tables]
        matched = _greedy_match(gt_boxes, pred_boxes, iou_thr)

        stats["gt_tables"] += len(gt_tables)
        stats["extra_pred"] += max(0, len(pred_tables) - len(matched))

        doc_pairs = []
        for gi, (gt_html, gb) in enumerate(gt_tables):
            if gi in matched:
                stats["matched"] += 1
                gt_table_list.append(gt_html)
                pred_table_list.append(pred_tables[matched[gi]][0])
                if match_dump_path:
                    pj = matched[gi]
                    doc_pairs.append({
                        "gt_idx": gi, "pred_idx": pj,
                        "iou": round(_iou(gb, pred_boxes[pj]), 4)
                        if gb and pred_boxes[pj] else None,
                        "gt_html_head": (gt_html or "")[:120],
                        "pred_html_head": (pred_tables[pj][0] or "")[:120],
                        "pred_norm_sha1": _table_content_sha1(pred_tables[pj][0]),
                    })
            else:
                stats["unmatched_gt"] += 1
                if unmatched_gt == "zero":
                    gt_table_list.append(gt_html)
                    pred_table_list.append("")  # 빈 pred -> TEDS 0
                # 'skip' 이면 아무것도 추가하지 않음
                if match_dump_path:
                    doc_pairs.append({
                        "gt_idx": gi, "pred_idx": None, "iou": None,
                        "gt_html_head": (gt_html or "")[:120],
                        "pred_html_head": "",
                    })
        if match_dump_path and doc_pairs:
            extra = [pj for pj in range(len(pred_tables))
                     if pj not in set(matched.values())]
            match_dump[image_key] = {"pairs": doc_pairs, "extra_pred_idx": extra}

    if match_dump_path:
        import json as _json
        with open(match_dump_path, "w") as f:
            _json.dump(match_dump, f, ensure_ascii=False, indent=1)
        print(f"[bbox-match-dump] {len(match_dump)} docs -> {match_dump_path}")

    return gt_table_list, pred_table_list, stats


def extract_tables(data : dict, normalize_header_tags: bool = False) -> str:
    """Extract tables from the dictionary data.

    Args:
        data (dict): The data to extract tables from.
        normalize_header_tags (bool): Whether to normalize th/td tags in extracted tables.

    Returns:
        str: The extracted tables from the data and a boolean indicating if the data has a table.
    """

    # return as is if data is a string
    html = '<html><body>'
    for elem in data['elements']:
        if elem['category'].lower() == 'table':
            table_html_elements = get_table_contents(elem['content']['html'])

            for table_html in table_html_elements:
                if normalize_header_tags:
                    table_html = normalize_table_header_tags(table_html)
                html += f'<table>{table_html}</table>'

    html += '</body></html>'

    return html


def has_table_content(html_data : str) -> bool:
    """Check if the table has content between <html><body> and </body></html>.

    Args:
        html_data (str): The html data to check.
    Returns:
        bool: True if the table has content, False otherwise
    """
    has_content = True
    if html_data.replace('<html><body>', '').replace('</body></html>', '') == '':
        has_content = False

    return has_content


def prepare_table_dataset(
    gt_data,
    pred_data,
    normalize_pred_table_header_tags: bool = True,
):
    """Prepare the tables for evaluation.
    Args:
        gt_data (dict): The ground truth dataset to evaluate.
        pred_data (dict): The predicted dataset to evaluate.
        normalize_pred_table_header_tags (bool): Whether to normalize th/td tags
            in predicted table HTML before evaluation.

    Returns:
        tuple (list, list): The list of ground truth and predicted tables.
    """

    gt_table_list = []
    pred_table_list = []
    for image_key in gt_data.keys():

        gt_elem = gt_data.get(image_key)
        pred_elem = pred_data.get(image_key)

        gt_tables = extract_tables(gt_elem)
        pred_tables = extract_tables(
            pred_elem,
            normalize_header_tags=normalize_pred_table_header_tags,
        )

        if not has_table_content(gt_tables):
            continue

        gt_table_list.append(gt_tables)
        pred_table_list.append(pred_tables)

    return gt_table_list, pred_table_list


def calc_table_score(gt_string, pred_string, evaluator):
    """Calculate the table evaluation score between the gold and pred strings.

    Args:
        gt_string (str): The ground truth html string to compare.
        pred_string (str): The predicted html string to compare.
        evaluator (TEDS/TEDS-S): The TEDS/TEDS-S evaluator to use.
    Returns:
        float: The table evaluation score.
    """
    refined_pred = pred_string
    refined_gold = gt_string
    if pred_string.startswith('<table>') and pred_string.endswith('</table>'):
        refined_pred = '<html><body>' + pred_string + '</body></html>'
    elif not pred_string.startswith('<html><body><table>') and not pred_string.endswith('</table></body></html>'):
        refined_pred = '<html><body><table>' + refined_pred + '</table></body></html>'

    if gt_string.startswith('<table>') and gt_string.endswith('</table>'):
        refined_gold = '<html><body>' + gt_string + '</body></html>'
    elif not gt_string.startswith('<html><body><table>') and not gt_string.endswith('</table></body></html>'):
        refined_gold = '<html><body><table>' + refined_gold + '</table></body></html>'

    # remove thead and tbody
    for tok in ['<thead>', '</thead>', '<tbody>', '</tbody>']:
        refined_pred = refined_pred.replace(tok, '')
        refined_gold = refined_gold.replace(tok, '')

    score = evaluator.evaluate(refined_pred, refined_gold)

    return score


def _score_pairs(gt_table_list, pred_table_list, structure_only):
    """매칭쌍 리스트에 대해 TEDS(or TEDS-S) 점수 리스트를 계산한다.

    pred 가 빈 문자열('')이면 미검출(unmatched)로 보고 0 점 처리한다.
    """
    evaluator = TEDSEvaluator(structure_only=structure_only)
    scores = []
    for gt_table_elem, pred_table_elem in zip(gt_table_list, pred_table_list):
        if not pred_table_elem:
            scores.append(0.0)
            continue
        scores.append(calc_table_score(gt_table_elem, pred_table_elem, evaluator))
    return scores


def evaluate_table(
    gt : dict,
    pred : dict,
    normalize_pred_table_header_tags: bool = True,
    match_mode: str = "index",
    bbox_iou_thr: float = 0.5,
    bbox_unmatched_gt: str = "zero",
    gt_coord_space: str = "extent",
    pred_coord_space: str = "extent",
    images_dir: str = None,
    pred_grid: float = 1024.0,
    match_dump_path: str = None,
) -> tuple:
    """Evaluate the table of the gt against the pred.

    Args:
        gt (dict): The gt layout to evaluate.
        pred (dict): The pred layout to evaluate against.
        normalize_pred_table_header_tags (bool): Whether to normalize th/td tags
            in predicted table HTML before evaluation.
        match_mode (str): 'index'(현행, 문서별 첫 테이블) 또는
            'bbox'(개별 테이블을 IoU 로 매칭).
        bbox_iou_thr (float): bbox 모드에서 매칭으로 인정할 최소 IoU.
        bbox_unmatched_gt (str): bbox 모드에서 pred 미검출 GT 테이블 처리.
            'zero'=0점 처리(옵션 b), 'skip'=채점 제외(옵션 a).
        gt_coord_space / pred_coord_space (str): 좌표공간(COORD_SPACES).
        images_dir (str): image 공간 사용 시 페이지 크기를 읽을 이미지 폴더.
        pred_grid (float): grid 공간에서 나눌 그리드 크기(기본 1024).

    Returns:
        tuple(float, float): The TEDS and TEDS-S scores for the table evaluation.
    """

    if match_mode == "bbox":
        gt_table_list, pred_table_list, stats = prepare_table_dataset_bbox(
            gt,
            pred,
            normalize_pred_table_header_tags=normalize_pred_table_header_tags,
            iou_thr=bbox_iou_thr,
            unmatched_gt=bbox_unmatched_gt,
            gt_coord_space=gt_coord_space,
            pred_coord_space=pred_coord_space,
            images_dir=images_dir,
            pred_grid=pred_grid,
            match_dump_path=match_dump_path,
        )
        print(
            f"[bbox-match] iou_thr={bbox_iou_thr} unmatched_gt={bbox_unmatched_gt} "
            f"gt_space={gt_coord_space} pred_space={pred_coord_space} | "
            f"GT tables={stats['gt_tables']} matched={stats['matched']} "
            f"unmatched_gt={stats['unmatched_gt']} extra_pred={stats['extra_pred']} "
            f"(scored units={len(gt_table_list)})"
        )
        if stats["degenerate_pred_docs"]:
            print(
                f"[Warning] {stats['degenerate_pred_docs']} docs have predicted tables "
                "without bbox (e.g. HTML-format predictions). bbox matching requires "
                "coordinate-bearing predictions (JSON output)."
            )
        if stats.get("missing_image_docs"):
            print(
                f"[Warning] {stats['missing_image_docs']} docs had no image in "
                "--images_dir (page size unknown -> those tables can't be matched)."
            )
    else:
        gt_table_list, pred_table_list = prepare_table_dataset(
            gt,
            pred,
            normalize_pred_table_header_tags=normalize_pred_table_header_tags,
        )

    avg_teds_score = 0.0
    avg_teds_s_score = 0.0

    if len(gt_table_list) == 0:
        print('[Warning] No tables found in the ground truth dataset.')
    elif len(pred_table_list) == 0:
        print('[Warning] No tables found in the prediction dataset.')
    else:
        # TEDS-S: structure only
        teds_s_scores = _score_pairs(gt_table_list, pred_table_list, structure_only=True)
        avg_teds_s_score = sum(teds_s_scores) / len(teds_s_scores)

        # TEDS: structure + content
        teds_scores = _score_pairs(gt_table_list, pred_table_list, structure_only=False)
        avg_teds_score = sum(teds_scores) / len(teds_scores)

    return avg_teds_score, avg_teds_s_score
