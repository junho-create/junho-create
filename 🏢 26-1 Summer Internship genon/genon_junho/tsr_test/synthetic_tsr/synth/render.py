# synth/render.py
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import random
import datetime
from PIL import Image, ImageDraw, ImageFont

from .table_graph import TableGraph


@dataclass
class RenderedSample:
    image: Image.Image
    cell_polys: Dict[int, List[Tuple[float, float]]]   # cell_id -> polygon(4 points)
    ocr_boxes: List[Dict[str, Any]]                    # [{"bbox":[x1,y1,x2,y2], "text":..., "cell_id":...}]


# =========================================================
# Font / Grid utils
# =========================================================

def _load_font(font_paths: Optional[List[str]], rng: random.Random, size: int):
    if font_paths:
        tries = min(8, max(1, len(font_paths)))
        for _ in range(tries):
            p = rng.choice(font_paths)
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _cum_lines(start: float, length: float, weights: List[float]) -> List[float]:
    s = sum(weights) if sum(weights) > 0 else 1.0
    acc = 0.0
    lines = [start]
    for w in weights:
        acc += w
        lines.append(start + length * (acc / s))
    lines[-1] = start + length
    return lines


# =========================================================
# Text layout utils (clip-safe)
# =========================================================

def _wrap_by_width(text: str, font, max_w: float, draw: ImageDraw.ImageDraw) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    # 공백 많은 경우(영문/혼합/단어형): 단어 기준
    if text.count(" ") >= 2:
        words = text.split()
        if not words:
            return []
        lines: List[str] = []
        cur = words[0]
        for w in words[1:]:
            trial = cur + " " + w
            bbox = draw.textbbox((0, 0), trial, font=font)
            if (bbox[2] - bbox[0]) <= max_w:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        return lines

    # 공백 적은 경우(한글 문장): 문자 기준
    lines: List[str] = []
    cur = ""
    for ch in text:
        trial = cur + ch
        bbox = draw.textbbox((0, 0), trial, font=font)
        if (bbox[2] - bbox[0]) <= max_w or cur == "":
            cur = trial
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def _clip_by_height(lines: List[str], font, max_h: float, draw: ImageDraw.ImageDraw, spacing: int = 4) -> List[str]:
    if not lines:
        return []
    bbox = draw.textbbox((0, 0), lines[0], font=font)
    line_h = (bbox[3] - bbox[1]) + spacing
    max_lines = max(1, int(max_h // line_h))
    clipped = lines[:max_lines]
    if len(lines) > max_lines:
        clipped[-1] = clipped[-1].rstrip() + "…"
    return clipped


# =========================================================
# Diverse text generator
# =========================================================

def _rand_date(rng: random.Random) -> str:
    y = rng.randint(2022, 2027)
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)
    return f"{y}.{m:02d}.{d:02d}"


def _rand_date_range(rng: random.Random) -> str:
    y = rng.randint(2022, 2027)
    m1 = rng.randint(1, 12)
    d1 = rng.randint(1, 28)
    m2 = min(12, m1 + rng.randint(0, 2))
    d2 = rng.randint(1, 28)
    return f"{y}.{m1:02d}.{d1:02d} ~ {y}.{m2:02d}.{d2:02d}"


def _rand_money(rng: random.Random) -> str:
    v = rng.choice([30000, 50000, 60000, 80000, 120000, 150000, 200000, 350000])
    if rng.random() < 0.2:
        return f"{rng.choice([50, 60, 80, 100, 120])}USD"
    return f"{v:,}원"


def _rand_percent(rng: random.Random) -> str:
    return f"{rng.choice([3, 5, 10, 15, 20])}% 이내"


def _rand_days(rng: random.Random) -> str:
    return f"{rng.choice([1, 2, 3, 5, 7, 10, 14, 30])}일"


def _join_sentences(rng: random.Random, n: int) -> str:
    clauses = [
        "규정 제28조에 따른 기준에 따라 지급한다.",
        "허가받은 목적 이외의 사용을 금지한다.",
        "개인정보가 포함된 자료는 외부 반출이 제한된다.",
        "이용기간 경과 후 3개월 이내 결과물을 제출해야 한다.",
        "필요 시 증빙서류를 추가 제출할 수 있다.",
        "중복 지급은 불가하며, 초과분은 환수한다.",
        "해당 항목은 내부 지침 및 별표 기준을 따른다.",
        "담당자 확인 후 최종 승인된다.",
    ]
    rng.shuffle(clauses)
    return " ".join(clauses[:n])


def _header_pool(style: str) -> List[str]:
    if style == "matrix":
        return ["구분", "대상", "서류 및 필기전형", "면접전형", "3% 이내", "5%", "10%", "비고"]
    if style == "form":
        return ["기관명", "부서", "직위", "성명", "연락처", "이용목적", "이용기간", "비고"]
    if style == "leave":
        return ["종류", "구분", "사유", "기간", "비고"]
    return ["적용범위", "부양가족 구분", "월 지급액", "비고", "구분", "대상", "사유", "기간"]


def _category_pool(style: str) -> List[str]:
    if style == "matrix":
        return ["보훈대상자", "장애인", "중증 장애인", "북한이탈주민", "다문화가족", "자립준비청년", "경력단절여성"]
    if style == "leave":
        return ["청원휴가", "결혼", "사망", "자녀 돌봄(유급)", "가족돌봄(무급)", "질병", "예비군/민방위"]
    return ["일반직원", "해외근무자", "단기파견", "장기파견", "배우자", "자녀", "부모", "기타"]


def _form_bullets(rng: random.Random) -> str:
    bullets = [
        "□ 이용목적:",
        "□ 이용기간:",
        "□ 이용항목(항목이 많을 경우 별도 기재):",
        "□ 결과물 형태(현황표/그래프 등):",
        "□ 준수사항 확인(서명):",
    ]
    rules = [
        "○ 허가받은 내용 이외의 목적으로 이용 금지",
        "○ 알게 된 비밀사항을 누설하거나 목적 외 사용 금지",
        "○ 서약서에 서명한 자 외 열람/복사 금지",
        "○ 이용 종료 후 자료 즉시 반납 및 파기",
    ]
    if rng.random() < 0.55:
        return rng.choice(bullets) + " " + rng.choice(["내부검토", "보고서 작성", "통계분석", "민원처리", "감사대응"])
    return rng.choice(rules)


def _infer_role(style: str, header_rows: int, cell_r: int, cell_c: int, cols: int, cell_w: float) -> str:
    # header
    if cell_r < header_rows:
        return "header"

    # matrix style: 오른쪽 좁은 열들은 마커/퍼센트
    if style == "matrix":
        if cell_c >= 2:
            return "marker_or_percent"
        return "category"

    # form style: 체크박스/문장
    if style == "form":
        if cell_w > 0.45:  # 넓은 셀은 문단/체크박스
            return "form_long"
        return "form_short"

    # leave/policy: 첫 열은 category, 가운데 넓은 열은 paragraph
    if cell_c == 0:
        return "category"
    if cell_c == cols - 1:
        return "note"
    if cell_w > 0.40:
        return "paragraph"
    # 나머지: money/date/days/random short
    return "value"


def _generate_cell_text(style: str, rng: random.Random, role: str) -> str:
    if role == "header":
        return rng.choice(_header_pool(style))

    if role == "category":
        return rng.choice(_category_pool(style))

    if role == "marker_or_percent":
        # 매트릭스 표는 ○ 표시/퍼센트/빈값 섞기
        p = rng.random()
        if p < 0.45:
            return "○"
        if p < 0.70:
            return _rand_percent(rng)
        return ""  # 일부 공란도 자연스럽게

    if role == "value":
        p = rng.random()
        if p < 0.30:
            return _rand_money(rng)
        if p < 0.55:
            return _rand_days(rng)
        if p < 0.80:
            return _rand_date_range(rng)
        # 혼합
        return f"{_rand_money(rng)} / {_rand_percent(rng)}"

    if role == "note":
        return rng.choice(["-", "해당 없음", "추가서류 제출", "담당자 확인", "별표 기준", "중복 불가", "증빙 필요"])

    if role == "paragraph":
        # 길이 다양한 문단
        n = rng.choice([2, 3, 4])
        base = _join_sentences(rng, n)
        if rng.random() < 0.35:
            base = f"※ {base}"
        return base

    if role == "form_long":
        # 체크박스/준수사항 문장
        if rng.random() < 0.65:
            return _form_bullets(rng)
        return " ".join([_form_bullets(rng) for _ in range(rng.randint(2, 3))])

    if role == "form_short":
        return rng.choice(["홍길동", "김민수", "박지현", "총무과", "인사팀", "감사실", "사무관", "주무관", "02-123-4567"])

    # fallback
    return rng.choice(["-", "확인", "검토", "해당", "기타"])


def _make_ocr_chunks_from_lines(lines: List[str], rng: random.Random) -> List[str]:
    """
    셀 텍스트를 OCR 박스 여러 개로 만들기 위한 chunk 리스트.
    기본은 '줄 단위'로, 때때로 한 줄을 2조각으로 쪼개 many-to-one 강화를 한다.
    """
    chunks: List[str] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if rng.random() < 0.30 and len(ln) >= 6:
            # 한 줄을 2조각으로
            k = rng.randint(2, max(2, len(ln) - 2))
            chunks.append(ln[:k].strip())
            chunks.append(ln[k:].strip())
        else:
            chunks.append(ln)
    return [c for c in chunks if c]


# =========================================================
# Main render (merge-aware + diverse text)
# =========================================================

def render_table(tg: TableGraph, cfg, rng: random.Random) -> RenderedSample:
    W, H = cfg.img_size
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    x0, y0 = cfg.margin, cfg.margin
    x1, y1 = W - cfg.margin, H - cfg.margin
    table_w, table_h = x1 - x0, y1 - y0

    style = getattr(cfg, "table_style", "policy")
    header_rows = 2 if style == "matrix" else (0 if style == "form" else 1)

    # --- 비균등 row/col weights (문서 느낌)
    if style == "matrix":
        col_wts = [4.5, 2.2] + [1.2] * max(0, tg.cols - 2)
        row_wts = [(1.0 if r < header_rows else 1.1) for r in range(tg.rows)]
    elif style == "form":
        base = [3.5, 2.2, 2.2, 2.8]
        col_wts = (base + [2.0] * max(0, tg.cols - len(base)))[:tg.cols]
        row_wts = [(0.9 if r < 3 else (2.2 if rng.random() < 0.5 else 1.6)) for r in range(tg.rows)]
    elif style == "leave":
        base = [1.4, 1.4, 6.0, 1.8]
        col_wts = (base + [2.0] * max(0, tg.cols - len(base)))[:tg.cols]
        row_wts = [(1.0 if r < header_rows else (2.5 if rng.random() < 0.25 else 1.3)) for r in range(tg.rows)]
    else:  # policy
        if tg.cols >= 4:
            col_wts = ([2.2] + [3.0] * (tg.cols - 2) + [2.0])[:tg.cols]
        else:
            col_wts = [1.0] * tg.cols
        row_wts = [(1.1 if r < header_rows else (1.8 if rng.random() < 0.3 else 1.3)) for r in range(tg.rows)]

    x_lines = _cum_lines(x0, table_w, col_wts[:tg.cols])
    y_lines = _cum_lines(y0, table_h, row_wts[:tg.rows])

    # --- merge 내부선 차단 마스크
    blocked_v = [[False] * (tg.cols + 1) for _ in range(tg.rows)]
    blocked_h = [[False] * (tg.rows + 1) for _ in range(tg.cols)]
    for cell in tg.cells:
        for k in range(cell.c + 1, cell.c + cell.colspan):
            for r in range(cell.r, cell.r + cell.rowspan):
                blocked_v[r][k] = True
        for k in range(cell.r + 1, cell.r + cell.rowspan):
            for c in range(cell.c, cell.c + cell.colspan):
                blocked_h[c][k] = True

    # --- 테두리/선
    draw.rectangle([x0, y0, x1, y1], outline="black", width=cfg.outer_thickness)

    # vertical segments
    for k in range(1, tg.cols):
        xx = x_lines[k]
        seg_start = None
        for r in range(tg.rows):
            if blocked_v[r][k]:
                if seg_start is not None:
                    draw.line([xx, y_lines[seg_start], xx, y_lines[r]], fill="black", width=cfg.inner_thickness)
                    seg_start = None
            else:
                if seg_start is None:
                    seg_start = r
        if seg_start is not None:
            draw.line([xx, y_lines[seg_start], xx, y_lines[tg.rows]], fill="black", width=cfg.inner_thickness)

    # horizontal segments
    for k in range(1, tg.rows):
        yy = y_lines[k]
        seg_start = None
        for c in range(tg.cols):
            if blocked_h[c][k]:
                if seg_start is not None:
                    draw.line([x_lines[seg_start], yy, x_lines[c], yy], fill="black", width=cfg.inner_thickness)
                    seg_start = None
            else:
                if seg_start is None:
                    seg_start = c
        if seg_start is not None:
            draw.line([x_lines[seg_start], yy, x_lines[tg.cols], yy], fill="black", width=cfg.inner_thickness)

    # header separator
    if header_rows > 0 and header_rows < tg.rows:
        yy = y_lines[header_rows]
        draw.line([x0, yy, x1, yy], fill="black", width=cfg.header_sep_thickness)

    # --- text / ocr
    cell_polys: Dict[int, List[Tuple[float, float]]] = {}
    ocr_boxes: List[Dict[str, Any]] = []

    font_header = _load_font(cfg.font_paths, rng, size=16)
    font_body = _load_font(cfg.font_paths, rng, size=14)

    pad_x, pad_y = 6, 6
    spacing = 4

    # h_align = rng.choice(["middle", "top"])
    # v_align = rng.choice(["center", "left"])
    # =================================================
    # ⭐ 표 단위 데이터셀 정렬 (한 번만 결정)
    # =================================================
    data_v_align = rng.choice(["middle", "top"])     # 데이터셀 수직
    data_h_align = rng.choice(["center", "left"])    # 데이터셀 수평

    for cell in tg.cells:
        px0 = x_lines[cell.c]
        py0 = y_lines[cell.r]
        px1 = x_lines[cell.c + cell.colspan]
        py1 = y_lines[cell.r + cell.rowspan]

        cell_polys[cell.id] = [(px0, py0), (px1, py0), (px1, py1), (px0, py1)]

        # header는 empty 금지(자연스러움)
        is_header_cell = (cell.r < header_rows)
        if (not is_header_cell) and (rng.random() < cfg.empty_cell_ratio):
            continue

        cell_w = px1 - px0
        cell_h = py1 - py0
        safe_w = max(10.0, cell_w - 2 * pad_x)
        safe_h = max(10.0, cell_h - 2 * pad_y)

        role = _infer_role(style, header_rows, cell.r, cell.c, tg.cols, cell_w / table_w)
        font = font_header if is_header_cell else font_body

        raw = _generate_cell_text(style, rng, role)
        if not raw.strip():
            # marker_or_percent에서 빈값 나오면 빈셀 취급(텍스트 없음)
            continue

        # wrap + clip
        lines = _wrap_by_width(raw, font, safe_w, draw)
        lines = _clip_by_height(lines, font, safe_h, draw, spacing=spacing)
        if not lines:
            continue

        # =================================================
        # ⭐ 셀 타입 기반 정렬 결정
        # =================================================
        is_header_cell = (cell.r < header_rows)

        # 숫자/금액 셀 판단 (간단하지만 충분히 효과적)
        text_for_check = " ".join(lines)
        is_numeric_cell = any(ch.isdigit() for ch in text_for_check) and (
            "원" in text_for_check or "%" in text_for_check or "," in text_for_check
        )

        if is_header_cell:
            # 헤더셀: 수직(중,상), 수평(중,왼)
            h_align = rng.choice(["center", "left"])
            v_align = rng.choice(["middle", "top"])
        else:
            # 데이터셀: 표 단위 정렬 유지
            v_align = data_v_align
            if is_numeric_cell:
                h_align = "right"      # 숫자/금액셀은 우측 정렬
            else:
                h_align = data_h_align

        # 텍스트 블록 크기 계산
        line_boxes = [draw.textbbox((0, 0), ln, font=font) for ln in lines]
        line_heights = [(b[3] - b[1]) for b in line_boxes]
        block_h = sum(line_heights) + spacing * (len(lines) - 1)
        block_w = max((b[2] - b[0]) for b in line_boxes)

        # 수평 정렬
        if h_align == "left":
            tx = px0 + pad_x
        elif h_align == "center":
            tx = px0 + (cell_w - block_w) / 2
        else:  # right
            tx = px1 - pad_x - block_w

        # 수직 정렬
        if v_align == "top":
            ty = py0 + pad_y
        elif v_align == "middle":
            ty = py0 + (cell_h - block_h) / 2
        else:
            ty = py1 - pad_y - block_h

        # =================================================
        # ⭐ 수정 핵심: 실제 텍스트 위치 기준 draw + OCR bbox
        # =================================================
        cur_y = ty
        for ln, lh in zip(lines, line_heights):
            draw.text((tx, cur_y), ln, fill="black", font=font)

            tb = draw.textbbox((tx, cur_y), ln, font=font)
            j = float(getattr(cfg, "ocr_jitter_px", 1.5))
            bbox = [
                float(tb[0] + rng.uniform(-j, j)),
                float(tb[1] + rng.uniform(-j, j)),
                float(tb[2] + rng.uniform(-j, j)),
                float(tb[3] + rng.uniform(-j, j)),
            ]

            ocr_boxes.append({
                "bbox": bbox,
                "text": ln,
                "cell_id": cell.id,
                # 디버깅/분석용 (학습에는 영향 없음)
                "h_align": h_align,
                "v_align": v_align,
            })

            cur_y += lh + spacing

    return RenderedSample(img, cell_polys, ocr_boxes)
