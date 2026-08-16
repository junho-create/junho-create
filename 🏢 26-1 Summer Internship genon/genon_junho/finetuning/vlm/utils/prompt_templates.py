"""
프롬프트 템플릿 관리 (통합 단일 파이프라인 — Chandra OCR_LAYOUT HTML 출력)

설계 원칙
--------
- 테이블/레이아웃을 분기 없이 동일하게 취급한다. 입력 이미지가 크롭된 표든
  전체 문서 페이지든, 모델은 동일 프롬프트로 페이지의 모든 영역을 검출해
  **Chandra 스타일 HTML layout block**으로 출력한다.
- 출력 스키마(assistant target): Datalab Chandra(https://github.com/datalab-to/chandra)
  의 OCR_LAYOUT_PROMPT 그대로. 각 영역은 다음 형태의 div 이다::

    <div data-bbox="x0 y0 x1 y1" data-label="<Label>">
    ...content...
    </div>

  - data-bbox 는 0-1000 으로 정규화된 정수 좌표 "x0 y0 x1 y1".
  - data-label 은 아래 11개 카테고리 중 하나.
  - Table 영역의 content 는 표 구조 HTML(<table>...</table>).
  - 그 외 영역의 content 는 reading-order 텍스트(<p>...</p> 등).
- 프롬프트는 Chandra OCR_LAYOUT_PROMPT 를 **원문 그대로** 사용하되,
  라벨 목록만 기존 11개 카테고리로 교체한다(tags/attributes/guidelines 는 원문 유지).
- 프롬프트 스타일은 단 2종만 사용한다:
    - chandra_no_ocr   : 이미지만 입력
    - chandra_with_ocr : 이미지 + OCR 텍스트/좌표 입력(학습 시 50:50 mix)
"""

import random
from typing import Optional

# =============================================================================
# Prompt style identifiers (2종만 사용)
# =============================================================================
PROMPT_STYLE_CHANDRA_NO_OCR = "chandra_no_ocr"
PROMPT_STYLE_CHANDRA_WITH_OCR = "chandra_with_ocr"

PROMPT_STYLE_FALLBACK = PROMPT_STYLE_CHANDRA_WITH_OCR

VALID_PROMPT_STYLES = {
    PROMPT_STYLE_CHANDRA_NO_OCR,
    PROMPT_STYLE_CHANDRA_WITH_OCR,
}

# 과거 prompt_style 문자열 → 2종으로의 매핑(디스크에 남아있는 기존 데이터 호환용).
# 심볼/프롬프트는 모두 제거되었지만, 문자열 값은 정규화로 흡수한다.
_PROMPT_STYLE_ALIASES = {
    # canonical
    "chandra_no_ocr": PROMPT_STYLE_CHANDRA_NO_OCR,
    "chandra_with_ocr": PROMPT_STYLE_CHANDRA_WITH_OCR,
    # 짧은 별칭
    "with_ocr": PROMPT_STYLE_CHANDRA_WITH_OCR,
    "without_ocr": PROMPT_STYLE_CHANDRA_NO_OCR,
    "no_ocr": PROMPT_STYLE_CHANDRA_NO_OCR,
    "ocr": PROMPT_STYLE_CHANDRA_WITH_OCR,
    # 레거시 chandra (table 전용)
    "chandra_table_with_ocr": PROMPT_STYLE_CHANDRA_WITH_OCR,
    "chandra_table_without_ocr": PROMPT_STYLE_CHANDRA_NO_OCR,
    "chandra_without_ocr": PROMPT_STYLE_CHANDRA_NO_OCR,
    "chandra_table_complex_with_ocr": PROMPT_STYLE_CHANDRA_WITH_OCR,
    "chandra_table_complex_without_ocr": PROMPT_STYLE_CHANDRA_NO_OCR,
    "chandra_complex_with_ocr": PROMPT_STYLE_CHANDRA_WITH_OCR,
    "chandra_complex_without_ocr": PROMPT_STYLE_CHANDRA_NO_OCR,
    "table_with_ocr": PROMPT_STYLE_CHANDRA_WITH_OCR,
    "table_without_ocr": PROMPT_STYLE_CHANDRA_NO_OCR,
    # 레거시 통합(JSON/HTML) — layout 은 이미지 전용이었으므로 no_ocr 로 흡수
    "unified_table_with_ocr": PROMPT_STYLE_CHANDRA_WITH_OCR,
    "unified_table_without_ocr": PROMPT_STYLE_CHANDRA_NO_OCR,
    "unified_table": PROMPT_STYLE_CHANDRA_WITH_OCR,
    "unified_layout": PROMPT_STYLE_CHANDRA_NO_OCR,
    "unified_html_table_with_ocr": PROMPT_STYLE_CHANDRA_WITH_OCR,
    "unified_html_table_without_ocr": PROMPT_STYLE_CHANDRA_NO_OCR,
    "unified_html_table": PROMPT_STYLE_CHANDRA_WITH_OCR,
    "unified_html_layout": PROMPT_STYLE_CHANDRA_NO_OCR,
    "html_layout": PROMPT_STYLE_CHANDRA_NO_OCR,
    "unified_html": PROMPT_STYLE_CHANDRA_NO_OCR,
    "unified": PROMPT_STYLE_CHANDRA_NO_OCR,
    "layout": PROMPT_STYLE_CHANDRA_NO_OCR,
}

# =============================================================================
# System Prompt (단일)
# =============================================================================
SYSTEM_PROMPT = (
    "You are an expert document OCR and layout analysis model. "
    "You convert a document image into HTML layout blocks. "
    "Output only the HTML layout blocks, with no extra commentary and no markdown code fences."
)

# =============================================================================
# User Prompt — Chandra OCR_LAYOUT_PROMPT 원문 (labels 만 11개 카테고리로 교체)
# =============================================================================
# 기존에 쓰던 11개 카테고리(학습 GT 분포 기준). Chandra 원문의 라벨 목록만 이걸로 교체한다.
_LAYOUT_LABELS = """- Caption
- Footnote
- Formula
- List-item
- Page-footer
- Page-header
- Picture
- Section-header
- Table
- Text
- Title"""

# Chandra 의 PROMPT_ENDING (ALLOWED_TAGS / ALLOWED_ATTRIBUTES / Guidelines) — 원문 그대로.
_CHANDRA_PROMPT_ENDING = """
Only use these tags ['math', 'br', 'i', 'b', 'u', 'del', 'sup', 'sub', 'table', 'tr', 'td', 'p', 'th', 'div', 'pre', 'h1', 'h2', 'h3', 'h4', 'h5', 'ul', 'ol', 'li', 'input', 'a', 'span', 'img', 'hr', 'tbody', 'small', 'caption', 'strong', 'thead', 'big', 'code', 'chem'], and these attributes ['class', 'colspan', 'rowspan', 'display', 'checked', 'type', 'border', 'value', 'style', 'href', 'alt', 'align', 'data-bbox', 'data-label'].

Guidelines:
* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property. Describe in detail inside the div tag. Also convert charts to high fidelity data, and convert diagrams to mermaid.
* Forms: Mark checkboxes and radio buttons properly.
* Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
* Chemistry: Use <chem>...</chem> tags for chemical formulas with reactive SMILES.
* Lists: Preserve indents and proper list markers.
* Use the simplest possible HTML structure that accurately represents the content of the block.
* Make sure the text is accurate and easy for a human to read and interpret.  Reading order should be correct and natural.
""".strip()

# bbox_space 식별자: 좌표계 종류.
BBOX_SPACE_NORM = "norm_1000"      # 0-1000 정규화(기존/기본)
BBOX_SPACE_IMAGE_PX = "image_px"   # 원본 이미지 픽셀(e20~)

# 본문 첫 문단의 좌표계 안내 문장만 좌표계별로 분기한다(나머지는 동일).
_COORD_SENTENCE_NORM = "Bboxes are normalized 0-1000."
_COORD_SENTENCE_PX = (
    "Bboxes are absolute pixel coordinates in the original image, "
    "which is {width} pixels wide and {height} pixels tall."
)

# Chandra OCR_LAYOUT_PROMPT 원문 (labels 만 _LAYOUT_LABELS 로 교체).
# {coord_sentence} 자리에 좌표계 안내 문장을 끼워 넣는다.
_CHANDRA_PROMPT_BODY_TMPL = f"""
OCR this image to HTML, arranged as layout blocks.  Each layout block should be a div with the data-bbox attribute representing the bounding box of the block in x0 y0 x1 y1 format.  {{coord_sentence}} The data-label attribute is the label for the block.

Use the following labels:
{_LAYOUT_LABELS}

{_CHANDRA_PROMPT_ENDING}
""".strip()

# 하위호환: 기존 0-1000 본문 상수(다른 모듈에서 import 할 수 있으므로 유지).
_CHANDRA_PROMPT_BODY = _CHANDRA_PROMPT_BODY_TMPL.format(coord_sentence=_COORD_SENTENCE_NORM)

# with_ocr 일 때 본문 뒤에 덧붙이는 OCR 안내 블록. {coord_note} 로 좌표계 분기.
_CHANDRA_OCR_BLOCK = """
In addition to the image, you are provided with OCR text extracted from the same image.
Each OCR item includes recognized text and its bounding box, {coord_note}

<ocr_info>
{ocr_info}
</ocr_info>

Use the OCR text as the primary source of textual content, and use the image to resolve
the layout, region boundaries, labels, merged cells, reading order, and ambiguous cases.
""".strip()

_OCR_COORD_NOTE_NORM = "normalized to the same 0-1000 coordinate space."
_OCR_COORD_NOTE_PX = "in the same absolute pixel coordinate space of the {width}x{height} image."


def _render_prompt_body(
    bbox_scale: int = 1024,
    bbox_space: str = BBOX_SPACE_NORM,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> str:
    """Chandra OCR_LAYOUT 본문을 반환한다.

    bbox_space=image_px 이고 (image_width, image_height) 가 주어지면 좌표계 안내를
    '원본 이미지 픽셀 + 이미지 크기'로 바꾼다. 그 외에는 기존 0-1000 안내를 쓴다.
    """
    if bbox_space == BBOX_SPACE_IMAGE_PX and image_width and image_height:
        coord = _COORD_SENTENCE_PX.format(width=int(image_width), height=int(image_height))
    else:
        coord = _COORD_SENTENCE_NORM
    return _CHANDRA_PROMPT_BODY_TMPL.format(coord_sentence=coord)

# 빈 셀 출력 규칙(옵션 append)
EMPTY_CELL_PROMPT_INSTRUCTION_TEMPLATE = (
    "If a cell is empty, output {token} inside that cell."
)

# =============================================================================
# Thinking Chain Templates
# =============================================================================

THINKING_TEMPLATE = """1. 테이블 크기 분석: {num_rows}행 × {num_cols}열
2. 구조 영역 분석:
{region_analysis}
3. Span 셀 탐지:
{span_analysis}
4. HTML 구조 생성 계획:
{structure_plan}"""

SPAN_ANALYSIS_TEMPLATES = {
    "colspan": "   - ({row},{col_start})~({row},{col_end}): \"{content}\" → colspan={span_size}",
    "rowspan": "   - ({row_start},{col})~({row_end},{col}): \"{content}\" → rowspan={span_size}",
    "both": (
        "   - ({row_start},{col_start})~({row_end},{col_end}): \"{content}\" "
        "→ colspan={colspan}, rowspan={rowspan}"
    ),
    "none": "   - Span 셀 없음: 모든 셀이 1×1 크기",
}

REGION_ANALYSIS_TEMPLATE = """   - 헤더 영역: {header_info}
   - 바디 영역: {body_info}"""


# =============================================================================
# Public helpers
# =============================================================================


def get_system_prompt(idx: Optional[int] = None) -> str:
    """시스템 프롬프트 반환 (단일 통합 프롬프트). idx 는 하위호환용으로 무시한다."""
    return SYSTEM_PROMPT


def get_user_prompt(idx: Optional[int] = None) -> str:
    """기본 유저 프롬프트(이미지 전용 본문) 반환. idx 는 하위호환용으로 무시한다."""
    return _render_prompt_body()


def _append_empty_cell_instruction(
    prompt_text: str,
    include_empty_cell_instruction: bool = False,
    empty_cell_token: str = "__EMPTY__",
) -> str:
    """프롬프트 말미에 빈 셀 출력 규칙을 선택적으로 추가한다."""
    prompt = str(prompt_text or "").strip()
    if not include_empty_cell_instruction:
        return prompt

    token = str(empty_cell_token or "").strip() or "__EMPTY__"
    instruction = EMPTY_CELL_PROMPT_INSTRUCTION_TEMPLATE.format(token=token)
    if instruction in prompt:
        return prompt
    return f"{prompt}\n* {instruction}"


def _format_ocr_info_text(ocr_info: Optional[list]) -> str:
    """OCR 정보 리스트를 프롬프트용 텍스트로 포맷한다.

    중요 정책:
    - OCR 아이템 개수는 축소/절단하지 않는다.
    - 학습/평가 안정성 문제는 환경/인프라 설정으로 해결한다.
    """
    if not ocr_info:
        return "[]"

    lines = []
    for item in ocr_info:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        bbox = item.get("bbox", [])
        lines.append(f"- text: {text} | bbox: {bbox}")

    return "\n".join(lines) if lines else "[]"


def normalize_prompt_style(prompt_style: Optional[str]) -> str:
    """prompt_style 문자열을 2종 중 하나로 정규화한다. 미지원 값은 fallback."""
    style = str(prompt_style or PROMPT_STYLE_FALLBACK).strip().lower()
    style = _PROMPT_STYLE_ALIASES.get(style, style)
    if style not in VALID_PROMPT_STYLES:
        return PROMPT_STYLE_FALLBACK
    return style


def prompt_requires_ocr(prompt_style: str) -> bool:
    """해당 프롬프트 스타일이 OCR 정보를 요구하는지 반환."""
    return normalize_prompt_style(prompt_style) == PROMPT_STYLE_CHANDRA_WITH_OCR


def get_user_prompt_with_style(
    idx: Optional[int] = None,
    prompt_style: str = PROMPT_STYLE_CHANDRA_WITH_OCR,
    ocr_info: Optional[list] = None,
    bbox_scale: int = 1024,
    include_empty_cell_instruction: bool = False,
    empty_cell_token: str = "__EMPTY__",
    bbox_space: str = BBOX_SPACE_NORM,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> str:
    """
    스타일 기반 유저 프롬프트 반환 (테이블/레이아웃 공통).

    prompt_style:
    - chandra_with_ocr : 이미지 + OCR
    - chandra_no_ocr   : 이미지만

    bbox_space=image_px + (image_width, image_height) 이면 좌표계를 원본 픽셀로 안내한다.
    """
    prompt_style = normalize_prompt_style(prompt_style)
    is_px = bbox_space == BBOX_SPACE_IMAGE_PX and bool(image_width) and bool(image_height)
    body = _render_prompt_body(bbox_scale, bbox_space, image_width, image_height)

    if prompt_style == PROMPT_STYLE_CHANDRA_WITH_OCR:
        coord_note = (
            _OCR_COORD_NOTE_PX.format(width=int(image_width), height=int(image_height))
            if is_px else _OCR_COORD_NOTE_NORM
        )
        ocr_block = _CHANDRA_OCR_BLOCK.format(
            ocr_info=_format_ocr_info_text(ocr_info),
            coord_note=coord_note,
        )
        prompt_text = f"{body}\n\n{ocr_block}"
    else:
        prompt_text = body

    return _append_empty_cell_instruction(
        prompt_text=prompt_text,
        include_empty_cell_instruction=include_empty_cell_instruction,
        empty_cell_token=empty_cell_token,
    )


def build_thinking_chain(
    num_rows: int,
    num_cols: int,
    spans: list[dict],
    header_rows: int = 1,
) -> str:
    """
    Span 정보로부터 thinking chain 텍스트를 생성한다.

    Args:
        num_rows: 전체 행 수
        num_cols: 전체 열 수
        spans: span 정보 리스트. 각 항목은:
            {
                "row_start": int, "row_end": int,
                "col_start": int, "col_end": int,
                "content": str (optional)
            }
        header_rows: 헤더 행 수

    Returns:
        thinking chain 텍스트
    """
    # Region analysis
    header_info = f"0~{header_rows - 1}행 (총 {header_rows}행)"
    body_info = f"{header_rows}~{num_rows - 1}행 (총 {num_rows - header_rows}행)"
    region_analysis = REGION_ANALYSIS_TEMPLATE.format(
        header_info=header_info, body_info=body_info
    )

    # Span analysis
    if not spans:
        span_analysis = SPAN_ANALYSIS_TEMPLATES["none"]
    else:
        span_lines = []
        for sp in spans:
            rs, re = sp["row_start"], sp["row_end"]
            cs, ce = sp["col_start"], sp["col_end"]
            content = sp.get("content", "")[:20]  # 내용은 20자로 제한
            colspan = ce - cs + 1
            rowspan = re - rs + 1

            if rowspan > 1 and colspan > 1:
                template = SPAN_ANALYSIS_TEMPLATES["both"]
                line = template.format(
                    row_start=rs, col_start=cs,
                    row_end=re, col_end=ce,
                    content=content,
                    colspan=colspan, rowspan=rowspan,
                )
            elif colspan > 1:
                template = SPAN_ANALYSIS_TEMPLATES["colspan"]
                line = template.format(
                    row=rs, col_start=cs, col_end=ce,
                    content=content, span_size=colspan,
                )
            elif rowspan > 1:
                template = SPAN_ANALYSIS_TEMPLATES["rowspan"]
                line = template.format(
                    row_start=rs, row_end=re, col=cs,
                    content=content, span_size=rowspan,
                )
            else:
                continue

            span_lines.append(line)
        span_analysis = "\n".join(span_lines)

    # Structure plan
    structure_plan = (
        f"   - <table>: {num_rows}개 <tr>\n"
        f"   - Span 셀 수: {len(spans)}개\n"
        f"   - 헤더에 <thead> 사용, 바디에 <tbody> 사용"
    )

    return THINKING_TEMPLATE.format(
        num_rows=num_rows,
        num_cols=num_cols,
        region_analysis=region_analysis,
        span_analysis=span_analysis,
        structure_plan=structure_plan,
    )


def format_assistant_response(
    thinking: Optional[str],
    html: str,
    include_thinking: bool = True,
) -> str:
    """Thinking + HTML을 assistant 응답 포맷으로 결합한다.

    include_thinking=False 이거나 thinking이 비어있으면 HTML만 반환한다.
    """
    thinking_text = str(thinking or "").strip()
    if include_thinking and thinking_text:
        return f"<think>\n{thinking_text}\n</think>\n{html}"
    return html


def build_chat_messages(
    image_path: str,
    html: Optional[str] = None,
    thinking: Optional[str] = None,
    include_thinking: bool = True,
    prompt_idx: Optional[int] = None,
    prompt_style: str = PROMPT_STYLE_CHANDRA_WITH_OCR,
    ocr_info: Optional[list] = None,
    bbox_scale: int = 1024,
    include_empty_cell_prompt_instruction: bool = False,
    empty_cell_token: str = "__EMPTY__",
    bbox_space: str = BBOX_SPACE_NORM,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> list[dict]:
    """
    학습용 대화 메시지를 생성한다 (테이블/레이아웃 공통, 분기 없음).

    html=None이면 assistant turn을 포함하지 않는다 (inference용).

    Returns:
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": [{"type": "image", ...}, {"type": "text", ...}]},
            ({"role": "assistant", "content": "..."})  # html이 있을 때만 포함
        ]
    """
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": get_user_prompt_with_style(
                    idx=prompt_idx,
                    prompt_style=prompt_style,
                    ocr_info=ocr_info,
                    bbox_scale=bbox_scale,
                    include_empty_cell_instruction=include_empty_cell_prompt_instruction,
                    empty_cell_token=empty_cell_token,
                    bbox_space=bbox_space,
                    image_width=image_width,
                    image_height=image_height,
                )},
            ],
        },
    ]
    if html is not None:
        messages.append({
            "role": "assistant",
            "content": format_assistant_response(
                thinking=thinking,
                html=html,
                include_thinking=include_thinking,
            ),
        })
    return messages
