"""3모델 각자 학습/설계된 방식대로 프롬프트를 따로 준다 — 강제로 통일하면 성능이 깨진다.

    target/paddle : Chandra div-HTML 레이아웃 프롬프트 (train/vlm/utils/prompt_templates.py 원문)
    dots.ocr      : 공식 prompt_layout_all_en (모델 카드 그대로)
"""

from __future__ import annotations

CHANDRA_SYSTEM = (
    "You are an expert document OCR and layout analysis model. You convert a document "
    "image into HTML layout blocks. Output only the HTML layout blocks, with no extra "
    "commentary and no markdown code fences."
)

CHANDRA_USER_NO_OCR = """OCR this image to HTML, arranged as layout blocks.  Each layout block should be a div with the data-bbox attribute representing the bounding box of the block in x0 y0 x1 y1 format.  Bboxes are normalized 0-1000. The data-label attribute is the label for the block.

Use the following labels:
- Caption
- Footnote
- Formula
- List-item
- Page-footer
- Page-header
- Picture
- Section-header
- Table
- Text
- Title

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
* Make sure the text is accurate and easy for a human to read and interpret.  Reading order should be correct and natural."""

CHANDRA_OCR_SUFFIX = """

In addition to the image, you are provided with OCR text extracted from the same image.
Each OCR item includes recognized text and its bounding box, normalized to the same 0-1000 coordinate space.

<ocr_info>
{ocr_info}
</ocr_info>

Use the OCR text as the primary source of textual content, and use the image to resolve
the layout, region boundaries, labels, merged cells, reading order, and ambiguous cases."""

DOTS_OCR_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object."""


def build_messages(model_key: str, ocr_info: list | None = None) -> list[dict]:
    """model_key: "target" 은 Chandra 프롬프트, "external_a"(dots.ocr) 는 자체 프롬프트,
    "external_b"(paddle, PaddleOCR-VL-1.6-0.9B) 는 **텍스트 프롬프트 없이 이미지만** —
    이 모델은 PaddleOCRVL 파이프라인 안에서 크롭당 인식만 하도록 학습된 컴포넌트라 텍스트
    지시문을 주면 (Chandra 프롬프트든 평범한 "OCR this image"든) 반복루프/환각으로 무너짐.
    ocr_info 가 있으면 target 은 with-OCR 모드로 전환 (paddle/dots.ocr 은 프롬프트 고정)."""
    if model_key == "external_a":
        return [{"role": "user", "content": DOTS_OCR_PROMPT}]
    if model_key == "external_b":
        return [{"role": "user", "content": None}]  # client.py 가 텍스트 파트를 아예 생략함

    user = CHANDRA_USER_NO_OCR
    if ocr_info:
        import json as _json
        user = user + CHANDRA_OCR_SUFFIX.format(ocr_info=_json.dumps(ocr_info, ensure_ascii=False))
    return [
        {"role": "system", "content": CHANDRA_SYSTEM},
        {"role": "user", "content": user},
    ]
