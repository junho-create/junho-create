"""[2]/[4] CMCV: target/dots.ocr/paddle 3모델 추론 → 쌍별 교차동의도 → E/M/H 티어 + pseudo-label.

    python -m ocr_filter.cli cmcv run --limit 20 --workers 4

**티어는 쌍별(pairwise) 일치 여부로 정한다** — GT 유무와 무관하게 항상 계산 가능
(CMCV 원 설계 의도: GT 없는 새 데이터에서 여러 모델을 서로 대조해 신뢰도 추정). 세 쌍
(target-dots.ocr, target-paddle, dots.ocr-paddle) 각각을 `agree_min` 하나의 임계값으로
"일치/불일치" 이진 판정한 뒤:

    Easy   : target 이 dots.ocr 또는 paddle 중 **최소 하나와 일치** → 자동 사용
    Medium : target 은 둘 다와 불일치하지만 **dots.ocr-paddle 끼리는 일치**
             → 외부 합의본(dots.ocr 출력)을 pseudo-label 로 채택 (타겟의 약점 보완용, SFT 핵심)
    Hard   : 세 쌍 다 불일치 → 자동 라벨링 불가, hardcase(judge+refine+사람) 로

단순 "3쌍 평균 유사도"가 아니다 — 평균을 쓰면 "target 은 하나랑만 잘 맞고 나머지 둘은
안 맞는" Easy 케이스와 "외부 둘은 잘 맞는데 target 만 튀는" Medium 케이스를 둘 다 뭉개서
Hard 로 오분류하게 된다. `pairwise_scores`(참고용 평균)는 리포트/디버깅에만 남겨둔다.

`gt_score`/`teds_score`는 GT 있을 때만(레코드에 gt 없으면 둘 다 None) "이 판정이 실제로
맞았는지" 검증/캘리브레이션 용으로 같이 남긴다 — 티어 판정에는 전혀 안 씀.

쌍별 유사도는 텍스트 edit_distance 와 generic TEDS(구조+카테고리 인식 부분점수) 를 평균
낸 블렌드 값이다 — 셋 다 category+text 구조화 출력을 낸다: target/dots.ocr 은 원래
구조화 출력, paddle(PaddleOCR-VL-1.6) 도 마찬가지로 구조화 출력이 가능하지만
raw vLLM(이미지만, 프롬프트 없이) 로 부르면 레이아웃검출 없이 인식만 하는 0.9B
서브모듈 하나만 호출되어 순수 텍스트만 나온다 — 그래서 paddle 은 여기서 개별 호출이
아니라 실제 `PaddleOCRVL` 파이프라인(PP-DocLayoutV3 레이아웃검출 포함, `report/paddle_layout.py`)
을 **레코드 루프 시작 전에 전체 배치로 한 번** 호출해 구조화 결과를 미리 받아둔다
(report 갤러리와 동일한 방식 — 무거운 파이프라인 초기화를 1회로 상각).

레코드마다: target/dots.ocr 2모델만 호출(병렬, paddle 은 이미 배치로 계산됨) → normalize
→ 쌍별 동의도 → 티어+pseudo_label 배정 → 결과를 --out 에 한 줄씩 append
(재시작해도 이미 처리한 id 는 건너뜀).
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from ocr_filter.cmcv.client import call_model
from ocr_filter.cmcv.normalize import (
    full_text, normalize, normalize_category, table_htmls,
)
from ocr_filter.io.schema import Record, read_jsonl
from ocr_filter.metrics import SCORERS

Image.MAX_IMAGE_PIXELS = None  # 200dpi 원본 스캔 중 1억+ 픽셀짜리가 있어 PIL 기본 밤체크 비활성화

MODEL_KEYS = ("target", "external_a", "external_b")
CALLED_MODEL_KEYS = ("target", "external_a")  # external_b(paddle)는 배치 사전계산이라 여기 없음

# 모델별 bbox 좌표계 (normalize.py 파서 docstring 참고): target=Chandra 스펙 0-1000
# 정규화, dots.ocr/paddle=원본 이미지 픽셀. PageIoU 는 셋을 같은 [0,1] 좌표계로 맞춰야
# 비교 가능하다.
_BBOX_COORD_SYSTEM = {"target": "norm1000", "external_a": "pixel", "external_b": "pixel"}


def _image_size(image_path: str) -> tuple[int, int]:
    with Image.open(image_path) as im:
        return im.size


def _normalized_boxes(elements: list[dict], coord_system: str, w: int, h: int) -> list[list[float]]:
    boxes = []
    for e in elements:
        bbox = e.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        if coord_system == "norm1000":
            boxes.append([v / 1000 for v in bbox])
        else:
            x0, y0, x1, y1 = bbox
            boxes.append([x0 / w, y0 / h, x1 / w, y1 / h])
    return boxes

_write_lock = threading.Lock()


def _score_against_gt(
    record: Record, target_elements: list[dict],
) -> tuple[float | None, float | None]:
    """(gt_score, teds_bonus) — **참고/검증용**, 티어 판정에는 안 씀(agreement_score 가 함).
    gt 가 없는 레코드(미래의 GT 없는 신규 데이터)에서는 (None, None). gt_score 는 항상
    full_text 기반 edit_distance 라 paddle(구조 없는 순수 인식 텍스트) 포함 3모델 전부와
    공정하게 비교 가능. teds_bonus 는 표 레코드에서 target 이 실제 Table 블록을 냈을 때만
    채워지는 보너스 신호."""
    if not record.gt:
        return None, None
    if record.source_type == "table":
        gt_text = None
        htmls = table_htmls(target_elements)
        teds_bonus = SCORERS["teds"](record.gt, htmls[0]) if htmls else None
        gt_text = full_text([{"category": "Table", "text": record.gt}])
    else:
        gt_text = full_text(record.gt) if isinstance(record.gt, list) else str(record.gt or "")
        teds_bonus = None
    gt_score = SCORERS["edit_distance"](gt_text, full_text(target_elements))
    return gt_score, teds_bonus


def _elements_to_html(elements: list[dict]) -> str:
    """[{"category","text"}] → 대충 재구성한 HTML (generic TEDS 입력용). 카테고리를 **태그 이름
    자체**로 써야 `_rename_cost`(태그 비교)가 실제로 카테고리 일치/불일치를 반영한다 —
    `data-label` 속성으로만 넣으면 rename_cost 가 attrs 를 안 보므로 전부 <div> 로만 비교돼
    카테고리 차이가 무시된다. Table 은 원본 <table>...</table> 그대로 안에 넣는다."""
    parts = []
    for e in elements:
        text = e.get("text") or ""
        if e.get("category") == "Table":
            parts.append(text)  # 이미 <table>...</table> 원본 HTML 이라 그대로
        else:
            tag = (e.get("category") or "text").lower().replace(" ", "_")
            parts.append(f"<{tag}>{text}</{tag}>")
    return "".join(parts)


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = min(a[0], a[2]), min(a[1], a[3]), max(a[0], a[2]), max(a[1], a[3])
    bx0, by0, bx1, by1 = min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _match_by_bbox(
    a_items: list[tuple[dict, list[float]]], b_items: list[tuple[dict, list[float]]],
    iou_min: float = 0.3,
) -> tuple[list[tuple[dict, dict]], int, int]:
    """(element, 정규화 bbox) 목록 두 개를 bbox IoU 로 그리디 매칭.

    반환: (짝지어진 (a_el, b_el) 목록, a 미매칭 수, b 미매칭 수).

    **등장 순서 매칭을 대체한다.** 순서로 짝지으면 한쪽이 요소를 하나만 더/덜 잡아도
    그 뒤가 전부 한 칸씩 밀려 죄다 오답 처리된다 — 2026-07-30 실측에서 Table 슬롯의
    중앙값이 정확히 0.000 으로 나온 원인이 이것이었다(모델마다 표를 1~2개 다르게 잡음).
    bbox 로 짝지으면 개수가 달라도 실제로 같은 자리를 가리키는 것끼리 매칭된다."""
    pairs: list[tuple[dict, dict]] = []
    used_b: set[int] = set()
    cand = []
    for ia, (_, ba) in enumerate(a_items):
        for ib, (_, bb) in enumerate(b_items):
            iou = _bbox_iou(ba, bb)
            if iou >= iou_min:
                cand.append((iou, ia, ib))
    cand.sort(reverse=True)  # IoU 높은 짝부터 확정 (그리디)
    used_a: set[int] = set()
    for _iou, ia, ib in cand:
        if ia in used_a or ib in used_b:
            continue
        used_a.add(ia)
        used_b.add(ib)
        pairs.append((a_items[ia][0], b_items[ib][0]))
    return pairs, len(a_items) - len(used_a), len(b_items) - len(used_b)


def _matched_category_score(
    a_elements: list[dict], b_elements: list[dict], category: str, scorer,
) -> float | None:
    """`category` 요소끼리 등장 순서로 짝지어 `scorer(a_text, b_text)`의 평균을 낸다
    (Table→TEDS, Formula→CDM 라우팅용). 한쪽에만 있어 짝이 없는 요소는 0점 처리 —
    카테고리를 통째로 누락/환각한 것도 벌점을 받아야 pageIoU/전체텍스트에 묻혀 안 잡히는
    "표/수식만 통째로 빠짐" 케이스를 잡는다. `scorer`가 None(CDM 렌더 실패 등, 가짜 점수
    방지용 정책)을 내면 그 쌍만 집계에서 제외 — 전부 제외돼서 평균 낼 게 없으면 None(이
    슬롯 자체를 안 씀). 둘 다 이 카테고리가 없으면(n=0) 애초에 None."""
    a_texts = [e.get("text") or "" for e in a_elements if e.get("category") == category]
    b_texts = [e.get("text") or "" for e in b_elements if e.get("category") == category]
    n = max(len(a_texts), len(b_texts))
    if n == 0:
        return None
    scores = []
    for i in range(n):
        a_text = a_texts[i] if i < len(a_texts) else ""
        b_text = b_texts[i] if i < len(b_texts) else ""
        if not a_text or not b_text:
            scores.append(0.0)
            continue
        s = scorer(a_text, b_text)
        if s is not None:
            scores.append(s)
    return sum(scores) / len(scores) if scores else None


def _blended_pair_score(
    a_elements: list[dict], b_elements: list[dict],
    a_boxes: list[list[float]], b_boxes: list[list[float]],
) -> float:
    """텍스트 edit_distance / generic TEDS(구조+부분점수 텍스트) / PageIoU(레이아웃 커버리지)
    셋을 기본으로 1/3씩 평균. 순수 edit_distance 하나만 쓰면 레이아웃/카테고리는 사실상
    같은데 글자 몇 개 차이로 agree_min 바로 밑으로 떨어지는 경계 케이스가 잦았음(2026-07-13
    실측: target_paddle=0.835 처럼 agree_min=0.85 에 근소하게 못 미치는 사례들 — TEDS 는
    노드 단위로 부분점수를 줘서 이런 경우 좀 더 관대하게 나옴). PageIoU(2026-07-14 추가)는
    텍스트/구조가 아니라 bbox 배치 자체의 일치도를 보는 지표라 나머지 둘이 못 잡는
    "내용은 맞는데 박스 위치가 어긋난" 케이스를 보완한다.

    여기에 더해(2026-07-19, CDM 도입) **카테고리 라우팅** 슬롯을 조건부로 추가한다 — 논문
    (MinerU2.5-Pro CMCV) 이 text/table/formula 를 각각 edit distance/TEDS/CDM 으로 따로
    채점하는 걸 참고: 어느 한쪽이든 Table 요소가 있으면 `_matched_category_score`(TEDS)를,
    Formula 요소가 있으면 마찬가지로(CDM)를 슬롯에 추가한다. generic TEDS(위 구조 슬롯)는
    페이지 전체를 재구성한 근사 HTML 이라 표/수식 내부 구조까지 세밀하게 못 보는데, 이
    슬롯들은 실제 표 HTML/LaTeX 만 떼어 전용 지표로 채점한다. 표/수식이 없는 페이지는
    슬롯이 안 늘어나므로 기존 3-way 블렌드와 완전히 동일 — 기존 임계값 캘리브레이션 그대로
    유효하다."""
    text_score = SCORERS["edit_distance"](full_text(a_elements), full_text(b_elements))
    teds_score = SCORERS["generic_teds"](
        _elements_to_html(a_elements), _elements_to_html(b_elements),
    )
    iou_score = SCORERS["page_iou"](a_boxes, b_boxes)
    slots = [text_score, teds_score, iou_score]

    table_score = _matched_category_score(a_elements, b_elements, "Table", SCORERS["teds"])
    if table_score is not None:
        slots.append(table_score)

    formula_score = _matched_category_score(a_elements, b_elements, "Formula", SCORERS["cdm"])
    if formula_score is not None:
        slots.append(formula_score)

    return sum(slots) / len(slots)


# ── subtask 별 일관성 (MinerU2.5-Pro §3.2 "task-specific pairwise consistency metrics") ──
# 논문은 text/table/formula 를 **각각 다른 지표로 따로** 채점하고, subtask 마다 난이도
# 분포가 다르다고 명시한다("formula and table recognition are more sensitive to Hard
# samples, while text recognition benefits more from Medium samples"). 하나로 평균내던
# 기존 방식(_blended_pair_score)은 실측에서 두 가지 문제가 확인됐다 (2026-07-30):
#
#   1. 표가 있는 페이지일수록 텍스트의 발언권이 1/3 → 1/4 → 1/5 로 줄어든다. 표 페이지가
#      GT 품질 관점에서 오히려 더 까다로운데 방향이 반대다.
#   2. Table 슬롯 중앙값이 0.000 이라(순서매칭 어긋남, _match_by_bbox 주석 참고) 표
#      페이지의 블렌드를 거의 상수처럼 끌어내렸다.
#
# 그래서 subtask 별로 점수와 티어를 따로 낸다 — 한 페이지가 "텍스트는 Easy, 표는 Hard"
# 일 수 있고 그게 실제에 가깝다. layout 은 논문 Figure 1 의 클러스터 4분류(Layout/Text/
# Table/Formula)를 따라 별도 subtask 로 둔다.
SUBTASKS = ("layout", "text", "table", "formula")
_STRUCT_CATEGORIES = {"Table", "Formula"}  # text subtask 에서 제외 (전용 지표가 따로 있음)


def _items_with_boxes(
    elements: list[dict], coord_system: str, w: int, h: int,
) -> list[tuple[dict, list[float]]]:
    """(element, [0,1] 정규화 bbox) — bbox 없는 요소는 매칭 대상에서 제외."""
    out = []
    for e in elements:
        bbox = e.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        if coord_system == "norm1000":
            nb = [v / 1000 for v in bbox]
        else:
            nb = [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]
        out.append((e, nb))
    return out


def _category_subtask_score(
    a_items: list[tuple[dict, list[float]]], b_items: list[tuple[dict, list[float]]],
    category: str, scorer,
) -> float | None:
    """Table/Formula 처럼 전용 지표가 있는 subtask 를 bbox 매칭 기준으로 채점.

    양쪽 다 해당 카테고리가 없으면 None(이 페이지는 그 subtask 의 판정 대상이 아님).
    한쪽에만 있는(미매칭) 요소는 0점으로 집계한다 — 표/수식을 통째로 누락하거나
    환각한 것도 벌점을 받아야 한다. scorer 가 None(CDM 렌더 실패 등)을 내면 그 짝만 제외.

    카테고리 선별은 **정규화한 값**으로 한다 — paddle 은 표를 'table', 그림을 'chart' 로
    내는 식이라 원본 문자열로 거르면 그쪽 표/수식이 통째로 누락된 것처럼 잡힌다."""
    a_cat = [(e, b) for e, b in a_items if normalize_category(e.get("category") or "") == category]
    b_cat = [(e, b) for e, b in b_items if normalize_category(e.get("category") or "") == category]
    if not a_cat and not b_cat:
        return None
    pairs, a_miss, b_miss = _match_by_bbox(a_cat, b_cat)
    scores: list[float] = []
    for ea, eb in pairs:
        s = scorer(ea.get("text") or "", eb.get("text") or "")
        if s is not None:
            scores.append(s)
    scores.extend([0.0] * (a_miss + b_miss))
    return sum(scores) / len(scores) if scores else None


def _subtask_pair_scores(
    a_elements: list[dict], b_elements: list[dict],
    a_items: list[tuple[dict, list[float]]], b_items: list[tuple[dict, list[float]]],
    a_boxes: list[list[float]], b_boxes: list[list[float]],
) -> dict[str, float | None]:
    """subtask → 쌍 일관성 점수(해당 없으면 None).

        layout  : PageIoU (bbox 커버리지) — 박스 배치 자체의 일치도
        text    : edit distance (Table/Formula 를 뺀 본문 평문)
        table   : TEDS (bbox 매칭된 표끼리)
        formula : CDM  (bbox 매칭된 수식끼리)
    """
    a_text = [e for e in a_elements if normalize_category(e.get("category") or "") not in _STRUCT_CATEGORIES]
    b_text = [e for e in b_elements if normalize_category(e.get("category") or "") not in _STRUCT_CATEGORIES]
    text_score = (SCORERS["edit_distance"](full_text(a_text), full_text(b_text))
                  if (a_text or b_text) else None)
    return {
        "layout": SCORERS["page_iou"](a_boxes, b_boxes),
        "text": text_score,
        "table": _category_subtask_score(a_items, b_items, "Table", SCORERS["teds"]),
        "formula": _category_subtask_score(a_items, b_items, "Formula", SCORERS["cdm"]),
    }


def _tier_from_pair_scores(
    td: float | None, tp: float | None, dp: float | None, agree_min: float,
) -> str | None:
    """한 subtask 의 세 쌍 점수 → 티어. 판정 근거가 없으면(전부 None) None.

    티어 정의는 논문과 동일하게 **target 을 기준으로** 한다:
      Easy   target 이 dots/paddle 중 최소 하나와 일치
      Medium target 은 둘 다와 불일치인데 외부 둘끼리는 일치 (외부 합의 = pseudo-label)
      Hard   셋 다 불일치"""
    if td is None and tp is None and dp is None:
        return None
    if (td is not None and td >= agree_min) or (tp is not None and tp >= agree_min):
        return "Easy"
    if dp is not None and dp >= agree_min:
        return "Medium"
    return "Hard"


_PAIRS = (("target_dots", "target", "external_a"),
          ("target_paddle", "target", "external_b"),
          ("dots_paddle", "external_a", "external_b"))


def _subtask_scores_and_tiers(
    elements_by_model: dict[str, list[dict]], image_path: str, agree_min: float,
) -> tuple[dict[str, dict[str, float | None]], dict[str, str | None]]:
    """subtask × 쌍 점수표와 subtask 별 티어를 함께 낸다.

    반환 예:
        ({"text": {"target_dots": 0.95, ...}, "table": {...}, ...},
         {"layout": "Easy", "text": "Easy", "table": "Hard", "formula": None})
    """
    any_bbox = any(e.get("bbox") for els in elements_by_model.values() for e in els)
    w, h = _image_size(image_path) if any_bbox else (1, 1)
    boxes = {k: _normalized_boxes(elements_by_model[k], _BBOX_COORD_SYSTEM[k], w, h)
             for k in MODEL_KEYS}
    items = {k: _items_with_boxes(elements_by_model[k], _BBOX_COORD_SYSTEM[k], w, h)
             for k in MODEL_KEYS}

    scores: dict[str, dict[str, float | None]] = {s: {} for s in SUBTASKS}
    for pair_name, ka, kb in _PAIRS:
        per = _subtask_pair_scores(
            elements_by_model[ka], elements_by_model[kb],
            items[ka], items[kb], boxes[ka], boxes[kb],
        )
        for s in SUBTASKS:
            scores[s][pair_name] = per[s]

    tiers = {
        s: _tier_from_pair_scores(
            scores[s]["target_dots"], scores[s]["target_paddle"], scores[s]["dots_paddle"],
            agree_min)
        for s in SUBTASKS
    }
    return scores, tiers


def _page_tier(subtask_tiers: dict[str, str | None]) -> str:
    """페이지 단위로 하나의 티어가 필요할 때(export/DDAS 호환) — **가장 나쁜 subtask** 를 따른다.
    표 하나가 Hard 인 페이지를 통째로 Easy 라고 부르면 그 표가 무검증 GT 가 되기 때문."""
    present = [t for t in subtask_tiers.values() if t]
    for worst in ("Hard", "Medium", "Easy"):
        if worst in present:
            return worst
    return "Hard"


def _pairwise_scores(elements_by_model: dict[str, list[dict]], image_path: str) -> dict[str, float]:
    """세 쌍 각각의 유사도. 셋 다 이제 구조화된(category+text+bbox) 출력을 낸다 — paddle 은
    cmcv 에서 원래 raw vLLM(이미지만, 프롬프트 없음)으로 불러 레이아웃 구조 없는 순수
    인식텍스트 한 덩어리만 나왔었는데(2026-07-13 최초 구현), 이는 PaddleOCR-VL-1.6 모델
    자체의 한계가 아니라 그 호출 방식의 한계였다 — report 갤러리는 처음부터 실제
    `PaddleOCRVL` 파이썬 파이프라인(PP-DocLayoutV3 레이아웃검출 포함, `report/paddle_layout.py`)
    을 배치로 호출해 category+bbox 있는 구조화 출력을 잘 뽑아왔다. cmcv 도 동일 파이프라인
    결과(`run_cmcv`가 사전에 배치로 계산해 `elements_by_model["external_b"]`에 넣어줌)를 쓰도록
    맞춰서, 세 쌍 전부 동일하게 텍스트+TEDS+PageIoU 블렌드로 비교한다. bbox 좌표계가
    모델마다 달라서(target=0-1000 정규화, dots.ocr/paddle=원본 픽셀) PageIoU 비교 전에
    image_path 로 페이지 크기를 읽어 셋 다 [0,1] 정규화 좌표로 맞춘다 (bbox 가 하나도
    없으면(예: 테스트용 가짜 엘리먼트) 이미지 파일 열기 자체를 생략)."""
    any_bbox = any(e.get("bbox") for els in elements_by_model.values() for e in els)
    w, h = _image_size(image_path) if any_bbox else (1, 1)
    norm_boxes = {
        key: _normalized_boxes(elements_by_model[key], _BBOX_COORD_SYSTEM[key], w, h)
        for key in MODEL_KEYS
    }
    return {
        "target_dots": _blended_pair_score(
            elements_by_model["target"], elements_by_model["external_a"],
            norm_boxes["target"], norm_boxes["external_a"]),
        "target_paddle": _blended_pair_score(
            elements_by_model["target"], elements_by_model["external_b"],
            norm_boxes["target"], norm_boxes["external_b"]),
        "dots_paddle": _blended_pair_score(
            elements_by_model["external_a"], elements_by_model["external_b"],
            norm_boxes["external_a"], norm_boxes["external_b"]),
    }


def _tier_and_pseudo_label(
    pairwise: dict[str, float], elements_by_model: dict[str, list[dict]], agree_min: float,
    text_min: float = 0.90,
) -> tuple[str, list[dict] | None, float | None]:
    """Easy: target 이 dots.ocr/paddle 중 최소 하나와 일치. Medium: target 은 둘 다와
    불일치하지만 dots.ocr-paddle 끼리는 일치 → dots.ocr 출력을 pseudo-label 로. Hard: 다 불일치.

    Medium 은 블렌드(`agree_min`)에 **추가로** 텍스트 전용 게이트(`text_min`)를 통과해야
    한다 — 블렌드는 3~5슬롯 중 레이아웃 전용 슬롯(page_iou/generic_teds)이 절반 이상이라
    "박스 배치는 같은데 글자를 다르게 읽은" 쌍이 0.85 를 넘길 수 있다. Medium 의
    pseudo-label 은 dots.ocr 텍스트가 GT 로 그대로 채택되므로 텍스트 자체의 상호일치를
    별도로 강제한다. 게이트에 걸리면 Hard 로 보낸다(rescue/judge 경로에서 요소 단위로 재검).

    반환: (tier, pseudo_label, dots_paddle_text_score) — 텍스트 점수는 Medium 후보였을
    때만 계산/기록(아니면 None), 캘리브레이션·리포트용."""
    if pairwise["target_dots"] >= agree_min or pairwise["target_paddle"] >= agree_min:
        return "Easy", None, None
    if pairwise["dots_paddle"] >= agree_min:
        text_score = SCORERS["edit_distance"](
            full_text(elements_by_model["external_a"]),
            full_text(elements_by_model["external_b"]),
        )
        if text_score >= text_min:
            return "Medium", elements_by_model["external_a"], text_score  # dots.ocr = 외부 합의본
        return "Hard", None, text_score  # 레이아웃은 합의, 텍스트는 불합의 → 텍스트 미검증 GT 차단
    return "Hard", None, None


def _process_one(
    record: Record, models_cfg: dict, agree_min: float, paddle_elements: list[dict],
    text_min: float = 0.90,
) -> dict:
    raw_by_model: dict[str, str] = {}
    errors: dict[str, str] = {}
    ocr_info = record.meta.get("ocr_info") if record.source_type == "table" else None

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(call_model, key, models_cfg[key], record.image_path, ocr_info): key
            for key in CALLED_MODEL_KEYS
        }
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                raw_by_model[key] = fut.result()
            except Exception as e:  # noqa: BLE001 — 모델 서버 에러는 기록만 하고 계속 진행
                raw_by_model[key] = ""
                errors[key] = f"{type(e).__name__}: {e}"

    elements_by_model = {k: normalize(k, raw_by_model.get(k, "")) for k in CALLED_MODEL_KEYS}
    elements_by_model["external_b"] = paddle_elements
    gt_score, teds_bonus = _score_against_gt(record, elements_by_model["target"])
    pairwise = _pairwise_scores(elements_by_model, record.image_path)  # 레거시 블렌드(리포트용)
    subtask_scores, subtask_tiers = _subtask_scores_and_tiers(
        elements_by_model, record.image_path, agree_min)
    tier = _page_tier(subtask_tiers)

    # pseudo-label 은 **text subtask 가 Medium 일 때**만 외부 합의본(dots.ocr)을 채택한다.
    # 페이지 티어가 Hard 여도(예: 표만 Hard) 본문 텍스트는 외부 합의가 성립할 수 있는데,
    # 그 경우 텍스트는 살리고 표만 hardcase 로 보내는 게 논문의 subtask 분리 취지다.
    pseudo_label = text_score = None
    if subtask_tiers["text"] == "Medium":
        text_score = SCORERS["edit_distance"](
            full_text([e for e in elements_by_model["external_a"]
                       if normalize_category(e.get("category") or "") not in _STRUCT_CATEGORIES]),
            full_text([e for e in elements_by_model["external_b"]
                       if normalize_category(e.get("category") or "") not in _STRUCT_CATEGORIES]),
        )
        if text_score >= text_min:
            pseudo_label = elements_by_model["external_a"]
        else:
            subtask_tiers["text"] = "Hard"  # 텍스트 미검증 GT 차단
            tier = _page_tier(subtask_tiers)

    return {
        "id": record.id,
        "source_type": record.source_type,
        "pairwise_scores": pairwise,
        "agreement_score": sum(pairwise.values()) / 3,  # 참고용 평균 (판정엔 안 씀)
        "subtask_scores": subtask_scores,  # {subtask: {pair: score|None}} — 논문식 task-specific
        "subtask_tiers": subtask_tiers,    # {subtask: Easy|Medium|Hard|None}
        "dots_paddle_text_score": text_score,  # text=Medium 후보였을 때만 (텍스트 게이트 판정값)
        "tier": tier,                  # 페이지 대표 티어 = 가장 나쁜 subtask
        "pseudo_label": pseudo_label,  # text subtask 가 Medium 일 때만 (dots.ocr 출력)
        "gt_score": gt_score,      # 참고/검증용 (GT 있을 때만) — 티어 판정엔 안 씀
        "teds_score": teds_bonus,  # 참고/검증용
        "n_elements": {k: len(v) for k, v in elements_by_model.items()},
        "elements": elements_by_model,  # 실제 채점에 쓴 3모델 출력 그대로 보존 — report 가
        # 갤러리 그릴 때 이걸 재사용해야 함(모델 재호출 금지). 재호출하면 vLLM
        # 배치/디코딩이 완전히 결정적이라는 보장이 없어(thinking 모델 특성상 특히) 갤러리에
        # 보이는 출력이 실제로 그 tier/score 를 만든 출력과 달라질 수 있다.
        "errors": errors or None,
    }


def run_cmcv(
    unified_path: str | Path,
    out_path: str | Path,
    models_cfg: dict,
    limit: int | None = None,
    source_type: str | None = None,
    ids: set[str] | None = None,
    workers: int = 4,
    agree_min: float = 0.85,
    text_min: float = 0.90,
) -> dict:
    """ids 를 주면 그 id 들만 (taster CMCV — 클러스터별 소량 샘플만 채점할 때 사용)."""
    from ocr_filter.report.paddle_layout import get_paddle_elements_batch  # 지연 임포트:
    # ocr_filter.report 패키지 __init__ 이 ocr_filter.cmcv.client 를 임포트해서 모듈
    # 최상단에 두면 순환임포트가 됨 (report -> cmcv.client, cmcv.run -> report).

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)["id"])

    records = list(read_jsonl(unified_path))
    if source_type:
        records = [r for r in records if r.source_type == source_type]
    if ids is not None:
        records = [r for r in records if r.id in ids]
    records = [r for r in records if r.id not in done_ids]
    if limit is not None:
        records = records[:limit]

    n_total, n_done, n_errors = len(records), 0, 0
    t0 = time.time()

    # paddle(external_b): 실제 PaddleOCRVL 파이프라인(레이아웃검출 포함)을 전체 배치로
    # 한 번에 호출 — report 갤러리와 동일한 방식(무거운 파이프라인 초기화를 1회로 상각).
    print(f"paddle 배치 실행 중... ({n_total}장)")
    paddle_by_path = get_paddle_elements_batch([r.image_path for r in records])
    print("paddle 배치 완료")

    with open(out_path, "a", encoding="utf-8") as out_f, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _process_one, r, models_cfg, agree_min, paddle_by_path.get(r.image_path, []),
                text_min,
            ): r
            for r in records
        }
        for fut in as_completed(futures):
            result = fut.result()
            with _write_lock:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
            n_done += 1
            if result["errors"]:
                n_errors += 1
            if n_done % 20 == 0 or n_done == n_total:
                elapsed = time.time() - t0
                print(f"[{n_done}/{n_total}] {elapsed:.0f}s 경과, "
                      f"errors={n_errors}, 최근 tier={result['tier']}")

    return {"n_total": n_total, "n_done": n_done, "n_errors": n_errors, "n_skipped": len(done_ids)}
