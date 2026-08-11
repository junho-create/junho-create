"""
프롬프트 템플릿 관리 (통합 단일 파이프라인 — JSON 레이아웃 디텍션)

설계 원칙
--------
- 테이블/레이아웃을 분기 없이 동일하게 취급한다. 입력 이미지가 크롭된 표든
  전체 문서 페이지든, 모델은 동일 프롬프트로 페이지의 모든 영역을 검출해
  **하나의 JSON 배열**(layout element 배열)로 출력한다.
- 출력 스키마(assistant target):
    [
      {"bbox": [x0, y0, x1, y1], "category": "<Category>", "text": "<content>"},
      ...
    ]
  - bbox 는 0-bbox_scale(기본 1024)로 정규화된 정수 좌표.
  - Table element 의 text 는 표 구조 HTML(<table>...</table>).
  - 그 외 element 의 text 는 reading-order 인식 텍스트.
- 프롬프트 스타일은 단 2종만 사용한다:
    - chandra_no_ocr   : 이미지만 입력
    - chandra_with_ocr : 이미지 + OCR 텍스트/좌표 입력
- Datalab Chandra OCR 프롬프트(https://github.com/datalab-to/chandra)의 흐름
  (문서 페이지로 취급 → 모든 영역 검출 → reading order)을 따르되, 출력은
  HTML 이 아닌 위 JSON 배열로 통일한다.
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
    "You are a document layout detection expert. "
    "You analyze a document image, detect every region, and output the result as a "
    "single JSON array of layout elements. "
    "Output only the JSON array, with no extra commentary and no markdown code fences."
)

# =============================================================================
# User Prompt (테이블 + 레이아웃 공통)
# =============================================================================
# Table element 의 text(표 구조 HTML)에 허용하는 태그/속성.
_CHANDRA_TABLE_TAGS = (
    "['table', 'thead', 'tbody', 'tr', 'td', 'th', 'caption', "
    "'span', 'br', 'b', 'i', 'u', 'sup', 'sub']"
)
_CHANDRA_TABLE_ATTRS = "['colspan', 'rowspan', 'style', 'align']"

# 검출 가능한 카테고리(학습 GT 분포 기준).
_CHANDRA_CATEGORIES = (
    '["Title", "Section-header", "Text", "List-item", "Caption", '
    '"Footnote", "Formula", "Page-header", "Page-footer", "Picture", "Table"]'
)

# 공통 본문: 입력을 항상 "문서 페이지"로 동일하게 취급한다(분기 없음).
# 페이지에 영역이 표 하나뿐이면 결과는 자연히 Table element 하나짜리 배열이 된다.
# {bbox_scale} 는 get_user_prompt_with_style 에서 .format 으로 채운다.
_CHANDRA_PROMPT_BODY_TEMPLATE = f"""
Detect the layout of this document image and output a JSON array, preserving the natural reading order.

Output a JSON array where each element is an object of the form:
{{{{"bbox": [x0, y0, x1, y1], "category": "<Category>", "text": "<content>"}}}}

Guidelines:
* Treat the image as a document page. Detect every region on the page and output one object per region in natural reading order. A page may contain a single region (for example, only a table) or many regions; handle both the same way.
* bbox is the region's bounding box [x0, y0, x1, y1] (top-left, bottom-right), normalized to a 0-{{bbox_scale}} coordinate space, as integers.
* category must be exactly one of: {_CHANDRA_CATEGORIES}.
* text is the textual content of the region:
  - For "Table", text is the table structure as HTML (a full <table>...</table>). Only use these tags {_CHANDRA_TABLE_TAGS} and these attributes {_CHANDRA_TABLE_ATTRS}, and use colspan/rowspan to represent merged cells.
  - For "Picture", text is an empty string ("").
  - For every other category, text is the recognized reading-order text of the region. Use \\n for line breaks within a region, and keep subscripts/superscripts and special characters.
* Order the elements by natural reading order (top-to-bottom, left-to-right; respect multi-column layouts).
* Output only the JSON array. Do not add any commentary, explanation, or markdown code fences.
""".strip()

# with_ocr 일 때 본문 뒤에 덧붙이는 OCR 안내 블록.
_CHANDRA_OCR_BLOCK = """
In addition to the image, you are provided with OCR text extracted from the same image.
Each OCR item includes recognized text and its bounding box, normalized to the same 0-{bbox_scale} coordinate space.

<ocr_info>
{ocr_info}
</ocr_info>

Use the OCR text as the primary source of textual content, and use the image to resolve
the layout, region boundaries, categories, merged cells, reading order, and ambiguous cases.
""".strip()


def _render_prompt_body(bbox_scale: int = 1024) -> str:
    """본문 템플릿에 bbox_scale 을 채워 반환한다."""
    return _CHANDRA_PROMPT_BODY_TEMPLATE.format(bbox_scale=int(bbox_scale or 1024))

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
) -> str:
    """
    스타일 기반 유저 프롬프트 반환 (테이블/레이아웃 공통).

    prompt_style:
    - chandra_with_ocr : 이미지 + OCR
    - chandra_no_ocr   : 이미지만
    """
    prompt_style = normalize_prompt_style(prompt_style)
    body = _render_prompt_body(bbox_scale)

    if prompt_style == PROMPT_STYLE_CHANDRA_WITH_OCR:
        ocr_block = _CHANDRA_OCR_BLOCK.format(
            bbox_scale=bbox_scale,
            ocr_info=_format_ocr_info_text(ocr_info),
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
