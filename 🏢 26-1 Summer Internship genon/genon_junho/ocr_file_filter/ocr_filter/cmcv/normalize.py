"""3모델 출력 → 공통 canonical form: list[{"category": str, "text": str}].

    target / paddle : Chandra div-HTML (`<div data-bbox="..." data-label="Table">...</div>`)
    dots.ocr        : JSON 배열 ([{"bbox":..., "category":..., "text":...}, ...])

파싱은 모델 생성 오류(코드펜스, 짝 안 맞는 태그, JSON 뒤에 잡담 등)에 관대하게 짠다 —
실패해서 빈 리스트를 내는 것보다, 최대한 뽑아내는 쪽이 CMCV 채점에 유리하다.
"""

from __future__ import annotations

import json
import re

_DIV_BLOCK_RE = re.compile(
    r'<div\s+data-bbox="([^"]*)"\s+data-label="([^"]*)"\s*>(.*?)</div>', re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
_CODE_FENCE_RE = re.compile(r"^```(?:json|html)?\s*|\s*```$", re.MULTILINE)

# 학습 타겟(gt_html)이 쓰는 표준 11종 카테고리(cmcv/prompts.py DOTS_OCR_PROMPT,
# hardcase/prompts.py CHANDRA_USER_NO_OCR 가 모델에게 지시하는 값과 동일).
STANDARD_CATEGORIES = {
    "Caption", "Footnote", "Formula", "List-item", "Page-footer", "Page-header",
    "Picture", "Section-header", "Table", "Text", "Title",
}


def _normkey(s: str) -> str:
    return re.sub(r"[-_ ]", "", s).strip().lower()


_CASE_KEY_TO_STANDARD = {_normkey(c): c for c in STANDARD_CATEGORIES}

# 표준에 없는 자유형식 레이블 -> 의미상 가장 가까운 표준 카테고리. 2026-07-20 실측
# (hardcase_judge, 즉 397B judge 모델 출력의 21%에서 발생 -- 프롬프트가 11종을 명시해도
# 큰 모델이 종종 다른 문자열을 냄, 9B target 은 SFT 로 거의 안 벗어남<0.3%) 기준으로
# 빈도 상위 레이블들을 수동 매핑. 여기 없는 나머지 롱테일(1~2건짜리 문서별 고유 레이블)은
# normalize_category() 가 Text 로 폴백한다.
_SEMANTIC_MAP_RAW = {
    # 참고문헌/각주/메타데이터류 -> Text
    "referencecontent": "Text", "reference": "Text", "referenceitem": "Text",
    "referencelist": "Text", "asidetext": "Text", "aside": "Text", "asideitem": "Text",
    "visionfootnote": "Footnote", "abstract": "Text", "authors": "Text",
    "authorlist": "Text", "author": "Text", "affiliation": "Text", "affiliations": "Text",
    "authoraffiliation": "Text", "keywords": "Text", "journalinfo": "Text",
    "publicationinfo": "Text", "dates": "Text", "citation": "Text", "citationbox": "Text",
    "copyright": "Text", "disclaimer": "Text", "note": "Text", "annotation": "Text",
    "metadata": "Text", "column": "Text", "textcolumn": "Text", "question": "Text",
    "answer": "Text", "form": "Text", "formsection": "Text", "inputfield": "Text",
    "email": "Text", "websiteurl": "Text", "heading": "Text", "content": "Text",
    "publishersnote": "Text", "sidebar": "Text", "sidebartab": "Text",
    "sidebarvertical": "Text", "sidebartitle": "Text", "sidebarmenu": "Text",
    "sidebartext": "Text", "link": "Text", "articlecard": "Text",
    "downloadinfo": "Text", "breadcrumb": "Text", "resultmessage": "Text",
    "timelineduration": "Text", "timelinepoint": "Text",
    # 그림/도표/아이콘/장식류 -> Picture
    "chart": "Picture", "diagram": "Picture", "icon": "Picture", "illustration": "Picture",
    "map": "Picture", "scheme": "Picture", "barcode": "Picture", "qrcode": "Picture",
    "logo": "Picture", "stamp": "Picture", "watermark": "Picture", "scale": "Picture",
    "signature": "Picture", "signatureblock": "Picture", "arrow": "Picture", "box": "Picture",
    "decoration": "Picture", "separator": "Picture", "panel": "Picture", "subpanel": "Picture",
    "figurepart": "Picture", "figurecomponent": "Picture", "emptypage": "Picture",
    "image": "Picture", "figure": "Picture", "button": "Picture", "buttongroup": "Picture",
    "qicon": "Picture", "aicon": "Picture", "titleicon": "Picture", "toolbar": "Picture",
    "dialog": "Picture", "dialogbox": "Picture", "statusbar": "Picture",
    "statusmessage": "Picture", "filterbuttons": "Picture", "actionarea": "Picture",
    "windowcontent": "Picture", "exceldownload": "Picture",
    # 페이지 번호류 -> Page-footer
    "pagenumber": "Page-footer", "page": "Page-footer", "linenumber": "Page-footer",
    "pagemarginnote": "Page-footer", "pagemargintext": "Page-footer",
    "pagemargin": "Page-footer", "footer": "Page-footer", "footertext": "Page-footer",
    "header": "Page-header",
    # 캡션/라벨류 -> Caption
    "figurelabel": "Caption", "sublabel": "Caption", "figurecaption": "Caption",
    "label": "Caption", "legend": "Caption", "tabletitle": "Caption",
    "tablecaption": "Caption",
    # 표 조각(구조 깨짐, div 하나가 표 전체가 아니라 셀/행 단위로 쪼개져 나온 경우) -> Table
    "tablecell": "Table", "tablerow": "Table", "tableheader": "Table",
    "tablefootnote": "Table", "tablefooter": "Table", "tablecontrols": "Table",
    "tablefilters": "Table",
    # 수식/코드류 -> Formula
    "displayformula": "Formula", "chem": "Formula", "algorithm": "Formula",
    # 리스트류
    "listgroup": "List-item",
}


def normalize_category(category: str) -> str:
    """모델이 낸 category 문자열을 표준 11종(STANDARD_CATEGORIES)으로 정규화.
    1) 이미 표준이면 그대로. 2) 케이스/구두점만 다르면(list-item vs List-item 등) 표준으로.
    3) 의미가 명확한 자유형식 레이블(_SEMANTIC_MAP_RAW)은 가장 가까운 표준으로.
    4) 그 나머지(문서별 1회성 레이블 등 롱테일)는 Text 로 폴백 — 학습 타겟의 레이블 어휘를
    항상 11종으로 고정해야 기존 레퍼런스 학습셋과 같은 SFT 런에 섞일 수 있다(2026-07-20,
    hardcase_judge 397B 출력에서 실측: 13%가 비표준 레이블)."""
    c = (category or "").strip()
    if c in STANDARD_CATEGORIES:
        return c
    key = _normkey(c)
    if key in _CASE_KEY_TO_STANDARD:
        return _CASE_KEY_TO_STANDARD[key]
    if key in _SEMANTIC_MAP_RAW:
        return _SEMANTIC_MAP_RAW[key]
    return "Text"


# ── 카테고리 합의 정책 (2026-07-30 확정) ──────────────────────────────────────
# **모델 출력 자체는 건드리지 않는다.** 각 모델의 원본 카테고리는 그대로 보존하고(리포트·
# 디버깅에 필요), 두 출력을 **비교할 때만** normalize_category() 를 태워 표준 11종 위에서 맞춘다.
# paddle(PP-DocLayoutV3)은 표준 밖 라벨을 꽤 낸다 — 250페이지 실측에서 reference_content 124,
# chart 98, vision_footnote 63, aside_text 53 = 338개. 이걸 비교 시점에 가까운 표준
# 카테고리로 흡수하지 않으면 dots.ocr 과 영원히 안 맞는다(Picture↔chart 등).
#
# 그 위에서 **엄격 합의가 필요한 카테고리**를 따로 둔다:
#
#     Picture / Table / Formula / Section-header
#
# 앞의 셋은 전용 검증 경로(렌더 대조·TEDS·CDM)를 타는 구조적 요소라 무엇인지 자체가 갈리면
# 검증이 성립하지 않는다. Section-header 는 본문 텍스트 성격이지만 문서 계층을 정의하므로
# 갈리면 아웃라인이 통째로 무너진다 — 실측에서도 텍스트 합의도가 33.4%로 최약체였다
# (Text 는 82.7%). 나머지(Text/List-item/Page-header/Page-footer/Caption/Footnote/Title …)는
# 전부 본문 텍스트 성격이라 서로 갈려도 크게 상관없다: dots 는 목록을 List-item 으로,
# paddle 은 Text 로 내는데(매칭 455쌍 전부 불일치, 혼동 1위 List-item→Text 444건) 이건
# 모델 어휘 차이지 실제 오류가 아니라서 이걸로 페이지를 탈락시키면 안 된다.
STRICT_CATEGORIES = {"Picture", "Table", "Formula", "Section-header"}


def category_agrees(a: str | None, b: str | None) -> bool:
    """두 모델의 카테고리를 합의된 것으로 볼지. 비교 전에 양쪽 다 표준 11종으로 정규화한다.

    엄격 카테고리(STRICT_CATEGORIES)는 정확히 같아야 하고, 그 외 본문 텍스트 성격끼리는
    서로 달라도 합의로 본다."""
    ca, cb = normalize_category(a or ""), normalize_category(b or "")
    if ca == cb:
        return True
    return ca not in STRICT_CATEGORIES and cb not in STRICT_CATEGORIES


def _strip_tags(html: str) -> str:
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", html)).strip()


def _strip_thinking(raw: str) -> str:
    """target(Qwen3.5) 은 thinking 모델이라 `<think>...</think>` 안에서 답을 미리
    "초안"으로 한 번 써보고 나서 `</think>` 뒤에 최종 답을 또 쓰는 경우가 흔하다
    (실측 확인: 같은 div 블록 세트가 사고 안/밖에 각각 한 번씩, 총 두 벌 등장 →
    이걸 안 걷어내고 파싱하면 요소 개수가 정확히 2배로 세져서 실제로는 정답인 페이지가
    "타겟이 중복 환각을 냈다"처럼 오판된다 — 모델 문제가 아니라 파서 문제였음, 2026-07-13
    실데이터로 재현·확인함). `</think>` 가 있으면 그 뒤쪽만, 없으면 원문 그대로."""
    if "</think>" in raw:
        return raw.rsplit("</think>", 1)[-1]
    return raw


def _parse_bbox_str(bbox_str: str) -> list[float] | None:
    """"x0 y0 x1 y1" (공백구분) → [x0,y0,x1,y1]. target 은 0-1000 정규화 좌표."""
    parts = bbox_str.split()
    if len(parts) != 4:
        return None
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


def parse_chandra_output(raw: str) -> list[dict]:
    """target/paddle 의 div-HTML 출력을 [{"category","text","bbox"}] 로.
    bbox 는 Chandra 스펙대로 0-1000 정규화 좌표 (렌더링할 땐 이미지 크기로 재환산 필요).
    category 는 normalize_category() 로 표준 11종으로 정규화 — 정규화를 먼저 하고 나서
    Table 분기를 판단해야 한다("table" 소문자처럼 표준과 케이스만 다른 라벨이 원래 코드처럼
    `label == "Table"` 로 비교하면 안 걸려서, 실제 표인데 원본 <table> HTML 이 아니라 태그가
    벗겨진 텍스트로 저장되는 버그가 있었다 — 2026-07-20 실데이터 2,309건에서 확인)."""
    text = _strip_thinking(raw or "")
    text = _CODE_FENCE_RE.sub("", text).strip()
    elements = []
    for bbox_str, label, inner in _DIV_BLOCK_RE.findall(text):
        inner = inner.strip()
        bbox = _parse_bbox_str(bbox_str)
        category = normalize_category(label)
        if category == "Table":
            elements.append({"category": category, "text": inner, "bbox": bbox})  # <table>...</table> 그대로
        else:
            elements.append({"category": category, "text": _strip_tags(inner), "bbox": bbox})
    return elements


def parse_dots_ocr_output(raw: str) -> list[dict]:
    """dots.ocr 의 JSON 배열 출력을 [{"category","text","bbox"}] 로. 코드펜스/잡담 섞여 나와도
    최선을 다해 첫 '[' 부터 마지막 ']' 까지만 잘라 파싱한다. bbox 는 원본 이미지 픽셀 좌표
    그대로(gt 와 같은 좌표계 — target 처럼 0-1000 정규화 아님, 실측 확인함)."""
    text = _strip_thinking(raw or "")
    text = _CODE_FENCE_RE.sub("", text).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        items = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return [
        {"category": normalize_category(item.get("category", "")), "text": item.get("text", ""),
         "bbox": item.get("bbox")}
        for item in items if isinstance(item, dict)
    ]


def parse_paddle_output(raw: str) -> list[dict]:
    """PaddleOCR-VL-1.6-0.9B 는 PaddleOCRVL 파이프라인 안에서 크롭당 인식만 하도록
    학습된 컴포넌트라, 우리가 직접 chat-completion 으로 부르면(텍스트 프롬프트 없이 이미지만
    줘야 정상 동작 — client.py/prompts.py 참고) 레이아웃 구조 없이 순수 인식 텍스트만 뱉는다.
    그래서 다른 두 모델처럼 category/bbox 구조를 못 만들고, 통째로 한 "Text" 블록(bbox=None)으로
    취급한다."""
    text = (raw or "").strip()
    return [{"category": "Text", "text": text, "bbox": None}] if text else []


PARSERS = {
    "target": parse_chandra_output,
    "external_a": parse_dots_ocr_output,
    "external_b": parse_paddle_output,
}


def normalize(model_key: str, raw: str) -> list[dict]:
    return PARSERS[model_key](raw)


def plain_text(elements: list[dict]) -> str:
    """전체페이지 본문 비교용 — Table 은 셀 텍스트만 뽑아 섞이지 않게 제외."""
    parts = [e["text"] for e in elements if e.get("category") != "Table" and e.get("text")]
    return "\n".join(parts)


def table_htmls(elements: list[dict]) -> list[str]:
    return [e["text"] for e in elements if e.get("category") == "Table" and e.get("text")]


def full_text(elements: list[dict]) -> str:
    """모델/포맷에 상관없이 쓸 수 있는 비교용 평문: Table 도 태그만 벗겨서 포함시킨다.
    구조 정보가 없는 paddle(순수 인식 텍스트)과, Table/Text 를 구조화해서 내놓는
    target/dots.ocr 을 같은 잣대(edit_distance)로 비교하려면 이 쪽을 써야 공정하다."""
    parts = []
    for e in elements:
        text = e.get("text") or ""
        if not text:
            continue
        parts.append(_strip_tags(text) if e.get("category") == "Table" else text)
    return "\n".join(parts)
