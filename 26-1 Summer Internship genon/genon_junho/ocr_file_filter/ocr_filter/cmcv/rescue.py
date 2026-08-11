"""[4.5] Hard 페이지에서 **요소 단위로** 합의된 부분만 건져내는 rescue (GPU 프리).

cmcv 는 페이지 하나에 티어 하나를 준다(정확히는 subtask 별로 주고, 대표 티어는 그중 가장
나쁜 것). 그래서 "본문 20블록은 두 외부 모델이 완벽히 일치하는데 표 하나만 갈렸다" 는
페이지가 통째로 Hard 로 떨어져 버려진다. 2026-07-30 실측(300건 샘플, agree_min=0.85):

    text  subtask   Easy 79.0%  Medium  8.0%  Hard 13.0%
    table subtask   Easy 20.0%  Medium  1.1%  Hard 78.9%   ← 병목
    페이지 대표     Easy 51.7%  Medium  8.3%  Hard 40.0%

text 가 Easy/Medium 인 페이지가 87% 인데 페이지 대표로는 60% 뿐 — 그 차이 27%p 가 전부
"표/수식 때문에 본문까지 같이 버려진" 페이지다. rescue 는 그 본문을 되찾는다.

방식: 두 **외부** 모델(dots.ocr, paddle) 출력을 겹치는 bbox 끼리 **N:M 그룹**으로 묶어서
(`_connected_groups`), 위치·카테고리·텍스트가 합의된 것만 남긴다. 남은 요소는 서로 독립적인
두 모델이 동의한 것이므로 target 의 개입 없이도 GT 로 쓸 수 있다(Medium 의 pseudo-label 과
같은 논리를 **요소 단위**로 적용). 합의 안 된 요소는 버리고, 너무 많이 버려서 페이지가
누더기가 되면(커버리지 미달) rescue 실패로 처리한다 — 라벨이 빠진 페이지는 어차피
후처리의 uncovered_ocr 필터에서 걸린다.

**왜 1:1 매칭이 아니라 N:M 그룹인가**: Hard 페이지 실측(2026-07-30)에서 dots.ocr↔paddle의
PageIoU(영역 커버리지) 중앙값은 0.897 로 이미 높은데, 요소 단위 1:1 IoU 매칭 회수율은
1%대였다. dots 가 문단 하나로 묶는 블록을 paddle 이 줄 단위 2~3개로 쪼개는 식의 분할 관성
차이 때문에 개별 짝이 IoU 임계값을 못 넘긴 것 — 겹치는 요소를 그룹으로 모아 텍스트를
이어붙여 비교하면 이 차이가 흡수되어 회수율이 22배(1.0%→22.0%, coverage_min=0.95) 뛴다.

**카테고리 합의 정책**: Picture/Table/Formula/Section-header 는 그룹 내에서도 엄격 1:1
매칭·합의를 요구한다(전용 검증 경로를 타거나 문서 계층을 정의하는 요소라 갈리면 안 됨).
나머지(List-item/Text/Caption/Page-header/... 등 본문 텍스트류)는 카테고리가 달라도
(dots=List-item, paddle=Text 등) 합의로 보고 그룹 전체를 텍스트 풀로 비교한다
(`ocr_filter.cmcv.normalize.STRICT_CATEGORIES`/`category_agrees` 참고).

**채택 출처**: category 태그는 **항상 dots.ocr(external_a) 기준**(11종 스키마로
파인튜닝된 모델이라 어휘가 GT 와 맞음). 콘텐츠(text/구조)는 Table 만 paddle 채택(셀
구조가 더 안정적, 셀 개행은 `clean_cell_newlines`로 제거), 그 외는 dots.ocr 채택.

모델 콜이 전혀 없다. cmcv_results.jsonl 에 이미 저장된 elements 만 재사용하므로 CMCV 가
도는 동안 CPU 에서 병행할 수 있다.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ocr_filter.cmcv.normalize import STRICT_CATEGORIES, full_text, normalize_category
from ocr_filter.cmcv.run import _BBOX_COORD_SYSTEM, _items_with_boxes, _match_by_bbox
from ocr_filter.metrics import SCORERS

# 카테고리별 텍스트 비교 지표 — subtask 판정과 같은 배정(논문 §3.2)을 요소 단위에 적용.
_CATEGORY_SCORER = {"Table": "teds", "Formula": "cdm"}
_DEFAULT_SCORER = "edit_distance"

# 합의된 요소를 어느 모델 출력으로 채택할지. 표는 paddle 이 셀 구조(rowspan/colspan)를
# 더 안정적으로 뽑아서 그쪽을 따르고, 나머지는 dots.ocr 을 따른다.
_ADOPT_FROM = {"Table": "b"}
_ADOPT_DEFAULT = "a"

# 텍스트가 비어 있어도 정상인 카테고리(그림 등)는 텍스트 비교를 건너뛰고 위치만 본다.
_TEXTLESS_CATEGORIES = {"Picture"}

_GRID = 250  # 커버리지 래스터 해상도. FILTER_LOGIC.md 의 uncovered_ocr 과 같은 값.


def clean_cell_newlines(html: str) -> str:
    """paddle 표 HTML 의 셀 안 개행을 제거한다.

    paddle 은 셀 안에서 줄이 바뀌면 실제 개행문자를 그대로 넣는다 — 실측 예:
        <td>4. 현재 투자하려는 자금의 투자 예정 기간을 선택해 주세요.\\n☐ 3년 이상\\n☐ 2년 이상…</td>
    이건 표의 논리적 구조가 아니라 원본의 시각적 줄바꿈일 뿐이고, 그대로 GT 에 넣으면
    모델이 셀 안에서 개행을 재현하도록 학습된다. 태그 사이 들여쓰기용 개행까지 같이
    사라지지만 HTML 의미에는 영향이 없다."""
    return (html or "").replace("\r\n", "").replace("\r", "").replace("\n", "")


def _adopted(ea: dict, eb: dict) -> tuple[dict, str]:
    """합의된 짝에서 채택본을 고른다. 표는 paddle, 나머지는 dots.ocr.

    **category 태그는 항상 dots.ocr(ea) 기준으로 낸다 — 텍스트/구조를 paddle 에서
    채택하는 경우(표)에도 마찬가지다.** dots.ocr 는 표준 11종 스키마로 파인튜닝된 모델이라
    카테고리 어휘가 이미 GT 와 맞고, paddle(PP-DocLayoutV3)은 그 스키마로 학습된 적이 없어
    chart/reference_content 같은 자유형식 라벨을 낸다(2026-07-30 실측, 250페이지에서 338개).
    표 하나를 예로 들면: 텍스트/rowspan 구조는 paddle 이 더 정확해 그쪽을 쓰지만, 그 표가
    "Table"이라는 사실 자체는 dots.ocr 쪽 라벨을 신뢰한다.

    `{**src, "category": category}`에서 `category` 가 뒤에 오므로, `src` 가 paddle 요소여서
    그 안에 파일 자체 라벨이 섞여 있어도(예: 'table' 소문자, 'chart') 항상 덮어써진다.

    반환: (채택 요소, "a"|"b") — 어느 쪽 콘텐츠를 채택했는지 같이 준다(카테고리는 항상
    "a" 기준이라도, bbox 는 채택본 것을 써야 해서 호출측이 콘텐츠 출처를 알아야 한다)."""
    category = normalize_category(ea.get("category") or "")  # 항상 dots.ocr(ea) 기준
    side = _ADOPT_FROM.get(category, _ADOPT_DEFAULT)
    src = eb if side == "b" else ea
    text = src.get("text") or ""
    if category == "Table":
        text = clean_cell_newlines(text)
    return {**src, "category": category, "text": text}, side


def _raster(boxes: list[list[float]], grid: int = _GRID) -> np.ndarray:
    """[0,1] 정규화 bbox 목록 → grid×grid 불리언 마스크(겹침 중복 안 셈)."""
    mask = np.zeros((grid, grid), dtype=bool)
    for b in boxes:
        if not b or len(b) != 4:
            continue
        x0, x1 = sorted((b[0], b[2]))
        y0, y1 = sorted((b[1], b[3]))
        gx0 = min(grid, max(0, int(x0 * grid)))
        gx1 = min(grid, max(gx0 + 1, int(np.ceil(x1 * grid))))
        gy0 = min(grid, max(0, int(y0 * grid)))
        gy1 = min(grid, max(gy0 + 1, int(np.ceil(y1 * grid))))
        mask[gy0:gy1, gx0:gx1] = True
    return mask


def _boxes_overlap(a: list[float], b: list[float]) -> bool:
    ax0, ax1 = sorted((a[0], a[2]))
    ay0, ay1 = sorted((a[1], a[3]))
    bx0, bx1 = sorted((b[0], b[2]))
    by0, by1 = sorted((b[1], b[3]))
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def _connected_groups(
    a_items: list[tuple[dict, list[float]]], b_items: list[tuple[dict, list[float]]],
) -> list[tuple[list[tuple[dict, list[float]]], list[tuple[dict, list[float]]]]]:
    """겹치는 박스끼리(a-b 뿐 아니라 a가 여러 b와, b가 여러 a와 겹치는 것도 전이적으로)
    하나의 그룹으로 묶는다 (Union-Find). 그룹 하나가 (a쪽 요소들, b쪽 요소들) N:M 대응.

    **1:1 IoU 매칭을 대체하는 핵심** — Hard 페이지 실측(2026-07-30)에서 dots.ocr↔paddle
    의 PageIoU(영역 커버리지) 중앙값이 0.897 인데도 요소 단위 1:1 매칭 회수율은 1%대였다.
    dots 가 문단 하나로 묶은 블록을 paddle 이 줄 단위 2~3개로 쪼개는 식의 분할 관성 차이
    때문에 IoU 임계값을 넘는 개별 짝이 안 나온 것 — 겹치는 요소를 그룹으로 모아 텍스트를
    이어붙여 비교하면 이 분할 차이가 흡수된다(같은 조건 재측정에서 회수율 22%로 22배 상승)."""
    na, nb = len(a_items), len(b_items)
    parent = list(range(na + nb))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i, (_, ba) in enumerate(a_items):
        for j, (_, bb) in enumerate(b_items):
            if _boxes_overlap(ba, bb):
                union(i, na + j)

    groups: dict[int, tuple[list, list]] = defaultdict(lambda: ([], []))
    for i, item in enumerate(a_items):
        groups[find(i)][0].append(item)
    for j, item in enumerate(b_items):
        groups[find(na + j)][1].append(item)
    return list(groups.values())


def _text_pool_agrees(
    a_list: list[tuple[dict, list[float]]], b_list: list[tuple[dict, list[float]]],
    text_min: float,
) -> tuple[bool, float | None]:
    """비엄격(본문 텍스트류) 요소 묶음끼리, 개별이 아니라 **그룹 전체를 이어붙여** 비교.
    dots 문단 1개 = paddle 라인 2~3개 같은 분할 차이를 흡수하기 위해서다."""
    ta = full_text([e for e, _ in a_list])
    tb = full_text([e for e, _ in b_list])
    if not ta and not tb:
        return True, None
    if not ta or not tb:
        return False, 0.0
    score = SCORERS[_DEFAULT_SCORER](ta, tb)
    return score >= text_min, score


def _text_agrees(a: dict, b: dict, text_min: float) -> tuple[bool, float | None]:
    """두 요소의 텍스트가 합의되는지. (합의여부, 점수) — 점수가 None 이면 비교 생략.

    지표 선택(Table→TEDS, Formula→CDM)은 **정규화한 카테고리**로 한다. paddle 이 표를
    'table' 로, 그림을 'chart' 로 내는 식이라 원본 문자열로 분기하면 표에 TEDS 가 아니라
    평문 편집거리가 걸린다."""
    category = normalize_category(a.get("category") or "")
    if category in _TEXTLESS_CATEGORIES:
        return True, None  # 그림류는 텍스트가 없는 게 정상 — 위치 합의만으로 충분
    ta = (a.get("text") or "").strip()
    tb = (b.get("text") or "").strip()
    if category == "Table":
        # 셀 안 개행은 표의 구조가 아니므로 비교 전에 양쪽 다 제거한다 — 안 그러면
        # "내용은 같은데 paddle 만 개행을 넣은" 표가 불합의로 잘못 걸러진다.
        ta, tb = clean_cell_newlines(ta), clean_cell_newlines(tb)
    if not ta and not tb:
        return True, None
    if not ta or not tb:
        return False, 0.0  # 한쪽만 글자를 읽었다 = 불합의
    score = SCORERS[_CATEGORY_SCORER.get(category, _DEFAULT_SCORER)](ta, tb)
    if score is None:
        # 전용 지표가 판정 불가(CDM 렌더 실패 등) → 평문 편집거리로 대체.
        # 여기서 그냥 통과시키면 미검증 텍스트가 GT 로 들어간다.
        score = SCORERS[_DEFAULT_SCORER](ta, tb)
    return score >= text_min, score


def rescue_page(
    a_elements: list[dict],
    b_elements: list[dict],
    img_size: tuple[int, int],
    a_coord: str = "pixel",
    b_coord: str = "pixel",
    iou_min: float = 0.5,
    text_min: float = 0.90,
    coverage_min: float = 0.90,
    require_same_category: bool = True,
) -> dict | None:
    """두 외부 모델 출력에서 합의된 요소만 남긴 부분 라벨을 만든다.

    반환 None = rescue 실패(건질 게 너무 적음). 성공 시:

        {"elements": [...], "coverage": 0.0~1.0, "n_accepted": int,
         "n_rejected": int, "reject_reasons": {...}}

    coverage 는 "두 모델이 잡은 전체 영역 중 합의된 요소가 덮는 비율"이다. 이게 낮으면
    페이지의 상당 부분이 라벨 없이 남는다는 뜻이고, 그런 페이지는 후처리(uncovered_ocr)에서
    어차피 탈락하므로 여기서 미리 끊는다.
    """
    w, h = img_size
    a_items = _items_with_boxes(a_elements, a_coord, w, h)
    b_items = _items_with_boxes(b_elements, b_coord, w, h)
    if not a_items or not b_items:
        return None

    accepted: list[dict] = []
    accepted_boxes: list[list[float]] = []
    reasons: dict[str, int] = {}
    box_of_a = {id(e): bb for e, bb in a_items}
    box_of_b = {id(e): bb for e, bb in b_items}

    def bump(key: str, n: int = 1) -> None:
        if n:
            reasons[key] = reasons.get(key, 0) + n

    for ga, gb in _connected_groups(a_items, b_items):
        if not ga or not gb:
            bump("unmatched", len(ga) + len(gb))
            continue

        if not require_same_category:
            # 카테고리 무시 — 그룹 전체를 하나의 텍스트 풀로 취급 (측정/디버깅용).
            ok, _ = _text_pool_agrees(ga, gb, text_min)
            if not ok:
                bump("text_mismatch", len(ga))
                continue
            for e, bb in ga:
                accepted.append({**e, "category": normalize_category(e.get("category") or "")})
                accepted_boxes.append(bb)
            continue

        def cat_of(item: tuple[dict, list[float]]) -> str:
            return normalize_category(item[0].get("category") or "")

        strict_a = {cat_of(it) for it in ga if cat_of(it) in STRICT_CATEGORIES}
        strict_b = {cat_of(it) for it in gb if cat_of(it) in STRICT_CATEGORIES}
        common_strict = strict_a & strict_b
        orphan_strict = (strict_a | strict_b) - common_strict

        if orphan_strict:
            # 한쪽에만 있는 엄격 카테고리(Picture/Table/Formula/Section-header) — 반대쪽에
            # 대응이 없다는 건 구조 자체가 다르게 인식됐다는 뜻이라 매칭 실패로 처리한다.
            n = sum(1 for it in ga if cat_of(it) in orphan_strict)
            n += sum(1 for it in gb if cat_of(it) in orphan_strict)
            bump("category_mismatch", n)

        # 엄격 카테고리는 카테고리끼리 그룹 내에서 세부 1:1 매칭 (보통 조각나지 않는다).
        for cat in common_strict:
            sub_a = [it for it in ga if cat_of(it) == cat]
            sub_b = [it for it in gb if cat_of(it) == cat]
            sub_pairs, miss_a, miss_b = _match_by_bbox(sub_a, sub_b, iou_min=iou_min)
            bump("unmatched", miss_a + miss_b)
            for ea, eb in sub_pairs:
                ok, _score = _text_agrees(ea, eb, text_min)
                if not ok:
                    bump("text_mismatch")
                    continue
                chosen, side = _adopted(ea, eb)
                accepted.append(chosen)
                accepted_boxes.append(box_of_b[id(eb)] if side == "b" else box_of_a[id(ea)])

        # 비엄격(본문 텍스트류) 요소는 그룹 전체를 이어붙여 한 번에 합의 판정 — dots 의
        # 문단 1개가 paddle 의 라인 2~3개에 대응하는 분할 차이를 여기서 흡수한다.
        nonstrict_a = [it for it in ga if cat_of(it) not in STRICT_CATEGORIES]
        nonstrict_b = [it for it in gb if cat_of(it) not in STRICT_CATEGORIES]
        if nonstrict_a or nonstrict_b:
            if not nonstrict_a or not nonstrict_b:
                bump("unmatched", len(nonstrict_a) + len(nonstrict_b))
            else:
                ok, _ = _text_pool_agrees(nonstrict_a, nonstrict_b, text_min)
                if not ok:
                    bump("text_mismatch", len(nonstrict_a))
                else:
                    for e, bb in nonstrict_a:
                        accepted.append({**e, "category": normalize_category(e.get("category") or "")})
                        accepted_boxes.append(bb)

    if not accepted:
        return None

    union = _raster([bb for _, bb in a_items] + [bb for _, bb in b_items])
    kept = _raster(accepted_boxes)
    union_area = int(union.sum())
    coverage = float((kept & union).sum() / union_area) if union_area else 0.0
    if coverage < coverage_min:
        return None

    return {
        "elements": accepted,
        "coverage": coverage,
        "n_accepted": len(accepted),
        "n_rejected": sum(reasons.values()),
        "reject_reasons": reasons,
    }


def run_rescue(
    cmcv_results_path: str | Path,
    out_path: str | Path,
    id_to_image_size,
    tiers: tuple[str, ...] = ("Hard",),
    iou_min: float = 0.5,
    text_min: float = 0.90,
    coverage_min: float = 0.90,
    limit: int | None = None,
) -> dict:
    """cmcv 결과에서 `tiers` 에 해당하는 레코드를 훑어 rescue 를 시도하고 결과를 append.

    id_to_image_size: id -> (w, h) 를 돌려주는 콜러블. cmcv 결과에는 image_path 가 없어서
    (unified.jsonl 에만 있다) 호출측이 매핑을 주입한다. 크기를 못 구하면 그 레코드는 스킵.

    이어하기 안전: out_path 에 이미 있는 id 는 건너뛴다(다른 스테이지와 동일한 관례).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}

    n_seen = n_rescued = n_failed = n_skipped = 0
    with open(cmcv_results_path, encoding="utf-8") as fin, \
            open(out_path, "a", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("tier") not in tiers or d["id"] in done:
                continue
            if limit is not None and n_seen >= limit:
                break
            n_seen += 1
            size = id_to_image_size(d["id"])
            if not size:
                n_skipped += 1
                continue
            els = d.get("elements") or {}
            result = rescue_page(
                els.get("external_a") or [], els.get("external_b") or [], size,
                a_coord=_BBOX_COORD_SYSTEM["external_a"], b_coord=_BBOX_COORD_SYSTEM["external_b"],
                iou_min=iou_min, text_min=text_min, coverage_min=coverage_min,
            )
            if result is None:
                n_failed += 1
                continue
            n_rescued += 1
            fout.write(json.dumps({
                "id": d["id"],
                "tier": "Medium-rescued",
                "label_source": "rescue_external_consensus",
                "label": result["elements"],
                "coverage": result["coverage"],
                "n_accepted": result["n_accepted"],
                "n_rejected": result["n_rejected"],
                "reject_reasons": result["reject_reasons"],
            }, ensure_ascii=False) + "\n")

    return {"n_seen": n_seen, "n_rescued": n_rescued, "n_failed": n_failed,
            "n_skipped": n_skipped,
            "rescue_rate": (n_rescued / n_seen) if n_seen else 0.0}
