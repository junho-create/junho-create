"""
GT layout bbox 를 PaddleOCR 조각들로 스냅 보정하는 프로토타입.

원본 train.jsonl 은 읽기 전용. 결과는 --out (새 파일) + --report 로만 저장.

매칭 원칙 (이전 설계 그대로):
  - IoU 아님. "포함율(containment)" = area(ocr ∩ gt)/area(ocr) 로 매칭.
  - 세로 y-band 게이트로 위/아래 줄 조각 오매칭 방지.
  - OCR 조각은 페이지 내 GT 박스에 1:1 greedy 배정.
가드 (오보정 방지):
  - 텍스트 보유 라벨만 보정 (Picture 등 제외).
  - robust extent (y outlier 제거 후 min/max).
  - 수축 방지: union 주축 스팬이 GT 스팬의 COVER_MIN 이상일 때만.
  - 이동량 캡: 각 변 이동 <= EDGE_CAP * gt_height, 면적비 [0.5,2.0], IoU(new,gt)>=IOU_MIN.
  - tighten 우선: 확장은 더 타이트한 캡.
"""
from __future__ import annotations
import argparse, html, json, re, statistics
from difflib import SequenceMatcher
from pathlib import Path

DIV_RX = re.compile(
    r'<div\b[^>]*\bdata-bbox="([^"]+)"[^>]*\bdata-label="([^"]+)"[^>]*>(.*?)</div>',
    re.DOTALL,
)

# 텍스트를 실제로 갖는 라벨만 보정 대상
TEXT_LABELS = {
    "Text", "List-item", "Section-header", "Page-footer", "Page-header",
    "Caption", "Title", "Footnote",
}
# Table 은 외곽 박스만(셀 bbox 없음), Formula/Picture 는 제외 — 실험에서 Table 포함 여부 토글
TABLE_LABELS = {"Table"}

CONTAIN_MIN = 0.6      # OCR 조각의 60% 이상이 GT 안에 들어와야 매칭
YBAND_MIN   = 0.5      # OCR 조각 높이의 50% 이상 y 겹침
COVER_MIN   = 0.6      # union 이 GT 주축 스팬의 60% 이상 덮어야 (수축 방지)
EDGE_CAP    = 1.0      # 각 변 이동 <= 1.0 * gt_height (tighten)
EDGE_CAP_EXP= 0.5      # 확장 방향은 더 보수적으로
IOU_MIN     = 0.5
AREA_LO, AREA_HI = 0.5, 2.0
VSHRINK_MIN = 0.6      # 세로 과수축 방지: 보정 높이가 GT 높이의 60% 미만이면 기각
                       # (OCR이 문단 일부 줄을 놓쳐 union이 짧아진 오보정 차단)

# --- loose 모드: 기하학 캡을 풀고 "텍스트 일치"로 안전장치를 대체 ---
CONTAIN_MIN_LOOSE = 0.35   # 뒤틀린 GT도 조각을 모을 수 있게 완화
YBAND_MIN_LOOSE   = 0.30
TSIM_MIN          = 0.55   # GT 텍스트 vs 매칭 OCR 텍스트 유사도 이상이면 큰 이동도 신뢰
IOU_TRUST         = 0.60   # 텍스트가 약해도 이 정도면 작은 변경이라 기하만으로 신뢰
AREA_LO_L, AREA_HI_L = 0.2, 5.0


def _clean_text(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    return re.sub(r"\s+", "", s)  # 공백 제거 후 비교(줄바꿈/띄어쓰기 차이 무시)


def text_sim(gt_text: str, ocr_items: list) -> float:
    a = _clean_text(gt_text)
    items = sorted(ocr_items, key=lambda t: (t[0][1], t[0][0]))  # bbox y0, x0 = 대략 읽기순
    b = _clean_text("".join(t[1] for t in items))
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def gt_recall(gt_text: str, ocr_items: list) -> float:
    """GT 글자 중 OCR 이 실제로 잡은 비율(재현율).
    유사도(ratio)와 달리 '부분만 잡힘'을 걸러낸다 — 매칭된 공통부분 길이 / GT 길이."""
    a = _clean_text(gt_text)
    items = sorted(ocr_items, key=lambda t: (t[0][1], t[0][0]))
    b = _clean_text("".join(t[1] for t in items))
    if not a:
        return 1.0          # GT 텍스트 없음(기호/빈 셀 등) → recall 게이트 스킵
    if not b:
        return 0.0
    matched = sum(bl.size for bl in SequenceMatcher(None, a, b).get_matching_blocks())
    return matched / len(a)


def parse_bbox(raw):
    parts = re.split(r"[,\s]+", raw.strip())
    try:
        v = [float(x) for x in parts[:4]]
    except ValueError:
        return None
    if len(v) != 4:
        return None
    x0, y0, x1, y1 = v
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def iou(a, b):
    i = inter(a, b)
    u = area(a) + area(b) - i
    return i / u if u > 0 else 0.0


def contain(ocr, gt):
    ao = area(ocr)
    return inter(ocr, gt) / ao if ao > 0 else 0.0


def yband(ocr, gt):
    h = ocr[3] - ocr[1]
    if h <= 0:
        return 0.0
    ov = min(ocr[3], gt[3]) - max(ocr[1], gt[1])
    return max(0.0, ov) / h


def robust_union(boxes):
    """순수 min/max extent. 매칭(포함율+y밴드)이 이미 조각들을 GT 영역으로 제한하므로
    별도 outlier 제거는 하지 않는다(예전 y-밴드/gap 필터는 여러 줄 블록·띄엄띄엄한
    x 조각을 정상인데도 잘라먹어 제거)."""
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def refine_box(gt, ocrs):
    """ocrs: 이 GT 에 배정된 조각들. (new_bbox 또는 None, reason)"""
    if not ocrs:
        return None, "no_match"
    u = robust_union(ocrs)
    if u is None:
        return None, "no_match"
    gw, gh = gt[2] - gt[0], gt[3] - gt[1]
    if gh <= 0 or gw <= 0:
        return None, "degenerate_gt"
    horiz = gw >= gh
    # 수축 방지: union 주축 스팬이 충분히 커야
    if horiz:
        if (u[2] - u[0]) < COVER_MIN * gw:
            return None, "cover_fail"
    else:
        if (u[3] - u[1]) < COVER_MIN * gh:
            return None, "cover_fail"
    # 이동량 캡 (변별로 tighten/expand 구분)
    for k, sign in ((0, -1), (1, -1), (2, +1), (3, +1)):
        # sign=+1 이면 좌표 증가가 확장, -1 이면 감소가 확장
        d = u[k] - gt[k]
        expand = (d * sign) > 0
        cap = (EDGE_CAP_EXP if expand else EDGE_CAP) * gh
        if abs(d) > cap:
            return None, "edge_cap"
    # 세로 과수축 방지 (가로 텍스트 블록에서 OCR 줄 누락 → 높이 붕괴 차단)
    if horiz and (u[3] - u[1]) < VSHRINK_MIN * gh:
        return None, "vshrink"
    ar = area(u) / area(gt) if area(gt) > 0 else 0
    if not (AREA_LO <= ar <= AREA_HI):
        return None, "area_ratio"
    if iou(u, gt) < IOU_MIN:
        return None, "iou_low"
    return [round(v, 1) for v in u], "ok"


NEI_OVERLAP_MAX = 0.15  # union 확장 후 이웃 GT 박스와의 추가 겹침 허용 상한(이웃면적 대비)


def refine_box_union(gt, ocrs, neighbors):
    """union(단조 확장) 방식: GT 를 OCR extent 로 감싸되, 이웃과 과하게 겹치면 기각.
    neighbors: 같은 페이지의 다른 GT 박스들."""
    if not ocrs:
        return None, "no_match"
    u = robust_union(ocrs)
    if u is None:
        return None, "no_match"
    cand = [min(gt[0], u[0]), min(gt[1], u[1]), max(gt[2], u[2]), max(gt[3], u[3])]
    if cand == gt:
        return None, "no_change"
    gh = gt[3] - gt[1] or 1
    # 확장량 캡(폭주 방지)
    for k in range(4):
        if abs(cand[k] - gt[k]) > EDGE_CAP_EXP * gh + 1:
            # 확장이 큰 변은 GT 로 되돌림(부분 확장 허용)
            cand[k] = gt[k]
    if cand == gt:
        return None, "edge_cap"
    # 이웃 과겹침 가드: 확장으로 새로 늘어난 겹침이 임계 초과면 기각
    for nb in neighbors:
        na = area(nb)
        if na <= 0:
            continue
        add = (inter(cand, nb) - inter(gt, nb)) / na
        if add > NEI_OVERLAP_MAX:
            return None, "nei_overlap"
    return [round(v, 1) for v in cand], "ok"


RECALL_STRONG = 0.90   # OCR 이 GT 텍스트 거의 다 잡음 → 완전 스냅(큰 뒤틀림 교정 허용)
RECALL_MED    = 0.60   # 일부만 잡음 → 확장만(안전), 수축 금지

# 선두 마커/기호/아이콘(불릿, 원문자, 괄호숫자, 대시, 여는대괄호 등) — OCR 이 자주 놓쳐
# 왼쪽 경계가 안으로 밀리며 잘리는 것을 막는다.
MARKER_RX = re.compile(
    r'^\s*(?:[①-⑳㉑-㉟⓪]|'                     # 원문자 숫자
    r'[Ⅰ-Ⅻⅰ-ⅻ]|'                              # 로마 숫자(섹션번호)
    r'[▪▶►•·○◦*✔√☑■□◆★☆]|'                    # 불릿/도형
    r'[→⇒➔➤☞▷]|'                               # 화살표
    r'[\[〈《「『<＜]|'                            # 여는 괄호/꺾쇠
    r'\(?\s*\d+\s*[\).]|'                        # 1) 1. (1)
    r'[가-힣]\)|[a-zA-Z]\)|[-–—]\s|[※])'         # 가) a) - 대시, ※
)
LEFT_INWARD_CAP = 0.75   # 마커 규칙에 안 걸려도 왼쪽 경계 안쪽 이동은 이 배수(gt높이)로 제한

# 문자/숫자/공백이 아닌 문자(*, ', -, 괄호, 마침표, 원문자 등 전부) — 위치 어디에
# 있든 OCR 이 놓치기 쉬워, 텍스트 흐름의 "교차축"(가로쓰기 문단이면 세로폭=높이,
# 세로쓰기 컬럼이면 가로폭=너비)은 이런 문자가 하나라도 있으면 축소를 금지한다.
# (선두 마커 보호는 긴축의 시작 경계만 다루므로 이것과 상호보완적)
SPECIAL_CHAR_RX = re.compile(r"[^0-9A-Za-z가-힣\s]")


def has_special_char(text: str) -> bool:
    plain = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    return bool(SPECIAL_CHAR_RX.search(plain))


def protect_cross_axis(gt, cand, gt_text):
    if not has_special_char(gt_text):
        return cand
    gw, gh = gt[2] - gt[0], gt[3] - gt[1]
    cand = list(cand)
    if gw >= gh:      # 가로쓰기 문단 → 세로폭(높이) 축소 금지
        cand[1] = min(cand[1], gt[1])
        cand[3] = max(cand[3], gt[3])
    else:             # 세로쓰기 컬럼 → 가로폭(너비) 축소 금지
        cand[0] = min(cand[0], gt[0])
        cand[2] = max(cand[2], gt[2])
    return cand


def _expand_only(gt, u):
    return [min(gt[0], u[0]), min(gt[1], u[1]), max(gt[2], u[2]), max(gt[3], u[3])]


def refine_box_loose(gt, ocr_items, gt_text):
    """신뢰도(=GT 텍스트 재현율)에 따라 보정 강도를 3단계로:
      recall>=0.9 → OCR extent 완전 스냅(큰 이동 허용)
      recall>=0.6 → 확장만(마커/끝 누락에 의한 수축 차단)
      그 외      → 작은 변경만 허용, 크면 기각."""
    if not ocr_items:
        return None, "no_match"
    boxes = [t[0] for t in ocr_items]
    u = robust_union(boxes)
    if u is None:
        return None, "no_match"
    ig = iou(u, gt)
    rec = gt_recall(gt_text, ocr_items)
    has_text = bool(_clean_text(gt_text))

    if has_text and rec >= RECALL_STRONG:
        cand = u                              # 전체 텍스트 확보 → 위치 신뢰, 완전 스냅
    elif has_text and rec >= RECALL_MED:
        cand = _expand_only(gt, u)            # 부분 확보 → 확장만(수축 금지)
    elif ig >= IOU_TRUST:
        cand = u                              # 텍스트 근거 약해도 변경이 작으면 허용
    else:
        return None, "text_weak"             # 크게 움직이는데 근거 부족 → 보류

    cand = list(cand)
    gh = gt[3] - gt[1] or 1
    # 선두 마커 보호: 왼쪽 경계를 안으로(오른쪽으로) 밀지 않음 (확장은 허용)
    gt_lead = html.unescape(re.sub(r"<[^>]+>", " ", gt_text or "")).lstrip()
    if MARKER_RX.match(gt_lead):
        cand[0] = min(cand[0], gt[0])
    else:
        # 범용 백스톱: 마커 미검출이어도 왼쪽 경계 안쪽 이동은 제한
        cand[0] = min(cand[0], gt[0] + LEFT_INWARD_CAP * gh)
    # 특수문자(문자/숫자/공백이 아닌 것)가 텍스트 어디에든 있으면 교차축은 축소 금지
    cand = protect_cross_axis(gt, cand, gt_text)
    cand = [round(v, 1) for v in cand]
    if cand == [round(v, 1) for v in gt]:
        return None, "no_change"
    ar = area(cand) / area(gt) if area(gt) > 0 else 0
    if not (AREA_LO_L <= ar <= AREA_HI_L):
        return None, "area_wild"
    return cand, "ok"


def process_record(rec, include_table=False, mode="snap"):
    gt_html = rec.get("gt_html", "")
    ocr = [o for o in (rec.get("ocr_info") or []) if o.get("bbox") and len(o["bbox"]) == 4]
    divs = []  # [span_start, span_end, bbox, label, inner_text]
    for m in DIV_RX.finditer(gt_html):
        bb = parse_bbox(m.group(1))
        if bb is None:
            continue
        divs.append([m.start(1), m.end(1), bb, m.group(2), m.group(3)])
    refinable_idx = [
        i for i, d in enumerate(divs)
        if d[3] in TEXT_LABELS or (include_table and d[3] in TABLE_LABELS)
    ]
    cmin = CONTAIN_MIN_LOOSE if mode == "loose" else CONTAIN_MIN
    ymin = YBAND_MIN_LOOSE if mode == "loose" else YBAND_MIN
    # 1:1 greedy 배정: 각 OCR 조각 → 포함율 최대 GT(대상 라벨) 에만
    assign = {i: [] for i in refinable_idx}
    for o in ocr:
        ob = [float(x) for x in o["bbox"]]
        best_i, best_c = None, cmin
        for i in refinable_idx:
            gt = divs[i][2]
            if yband(ob, gt) < ymin:
                continue
            c = contain(ob, gt)
            if c >= best_c:
                best_c, best_i = c, i
        if best_i is not None:
            assign[best_i].append((ob, o.get("text", "")))
    # 보정
    results = []  # (idx, old, new_or_None, reason)
    for i in refinable_idx:
        items = assign[i]
        boxes = [t[0] for t in items]
        if mode == "union":
            neighbors = [divs[j][2] for j in range(len(divs)) if j != i]
            new, reason = refine_box_union(divs[i][2], boxes, neighbors)
        elif mode == "loose":
            new, reason = refine_box_loose(divs[i][2], items, divs[i][4])
        else:
            new, reason = refine_box(divs[i][2], boxes)
        results.append((i, divs[i][2], new, reason))
    return divs, results


def bbox_to_str(b):
    return " ".join(str(int(round(v))) for v in b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True, help="보정 반영된 새 jsonl")
    ap.add_argument("--report", required=True)
    ap.add_argument("--examples", default="")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--include-table", action="store_true")
    ap.add_argument("--mode", choices=["snap", "union", "loose"], default="snap")
    ap.add_argument("--write-out", action="store_true", help="보정 jsonl 실제 기록")
    args = ap.parse_args()

    from collections import Counter
    reasons = Counter()
    shifts = []       # 채택된 박스의 최대 변 이동량 / gt_height
    direction = Counter()  # 채택된 변경의 방향(순확장/순수축/혼합)
    n_target = 0
    n_changed = 0
    examples = []

    fout = open(args.out, "w", encoding="utf-8") if args.write_out else None
    n = 0
    for line in open(args.inp, encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        divs, results = process_record(rec, include_table=args.include_table, mode=args.mode)
        new_html = rec.get("gt_html", "")
        # span 치환은 뒤에서부터
        repl = []
        for i, old, new, reason in results:
            n_target += 1
            reasons[reason] += 1
            if new is not None and new != old:
                gh = old[3] - old[1] or 1
                mshift = max(abs(new[k] - old[k]) for k in range(4)) / gh
                shifts.append(mshift)
                # 방향 분류: 각 변이 확장(밖으로) 인가 수축(안으로) 인가
                exp = any((new[k]-old[k])*s > 0.5 for k, s in ((0,-1),(1,-1),(2,1),(3,1)))
                shr = any((new[k]-old[k])*s > 0.5 for k, s in ((0,1),(1,1),(2,-1),(3,-1)))
                direction["expand_only" if exp and not shr else
                           "shrink_only" if shr and not exp else "mixed"] += 1
                n_changed += 1
                s, e = divs[i][0], divs[i][1]
                repl.append((s, e, bbox_to_str(new)))
                if len(examples) < 40 and reason == "ok":
                    txt = re.sub(r"<[^>]+>", "", "")  # placeholder
                    examples.append({
                        "image": rec.get("image_path", "")[-60:],
                        "label": divs[i][3],
                        "old": bbox_to_str(old),
                        "new": bbox_to_str(new),
                        "max_edge_shift_ratio": round(mshift, 3),
                    })
        if fout is not None:
            for s, e, val in sorted(repl, reverse=True):
                new_html = new_html[:s] + val + new_html[e:]
            rec["gt_html"] = new_html
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
        if n >= args.limit:
            break
    if fout:
        fout.close()

    lines = []
    lines.append(f"records processed : {n}")
    lines.append(f"target GT boxes   : {n_target}")
    lines.append(f"changed boxes     : {n_changed} ({n_changed/max(1,n_target)*100:.1f}% of targets)")
    lines.append("")
    lines.append(f"mode              : {args.mode}")
    lines.append("decision breakdown:")
    for k, v in reasons.most_common():
        lines.append(f"  {k:14s} {v:7d} ({v/max(1,n_target)*100:.1f}%)")
    lines.append("")
    lines.append("accepted change direction:")
    for k in ("expand_only", "shrink_only", "mixed"):
        v = direction.get(k, 0)
        lines.append(f"  {k:14s} {v:7d} ({v/max(1,n_changed)*100:.1f}%)")
    if shifts:
        lines.append("")
        lines.append("accepted edge-shift ratio (max edge move / gt_height):")
        lines.append(f"  median {statistics.median(shifts):.3f}  mean {statistics.mean(shifts):.3f}  "
                     f"p90 {sorted(shifts)[int(0.9*len(shifts))-1]:.3f}  max {max(shifts):.3f}")
    report = "\n".join(lines)
    print(report)
    Path(args.report).write_text(report + "\n", encoding="utf-8")
    if args.examples:
        Path(args.examples).write_text(
            json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
