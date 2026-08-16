
# chandra ocr prompt with ocr info
# ocr_info: list of dict with 'text' and 'bbox' keys
# example ocr_info:
# [{'text': '(적용사례) 자기신용위험의 적용사례로 회사채 스프레드를 적용 하거나, CDS 프리미엄을 자기신용위험으로 고려', 'bbox': [97, 103, 926, 154]}, {'text': '* 단, 직접적인 보험부채 공정가치 적용사례는 기준서상 포함되지 않음', 'bbox': [142, 163, 871, 182]}, ... ]
# bbox_scale: int, the scale to which bbox are normalized, e.g., 1024
prompt_user_with_ocr = """
OCR this image to HTML, arranged as layout blocks. Each layout block should be a div with the data-bbox attribute representing the bounding box of the block in [x0, y0, x1, y1] format. Bboxes are normalized to a 0-{bbox_scale} coordinate space. The data-label attribute is the label for the block.

In addition to the image, you are also provided with OCR text information extracted from the same page. Each OCR item includes the recognized text and its bounding box. The OCR bounding boxes are in the same normalized 0-{bbox_scale} coordinate space.

<ocr_info>
{ocr_info}
</ocr_info>

You MUST use BOTH:
- the visual information from the image
- the provided OCR text and bounding boxes

to accurately reconstruct the document content and layout.

The OCR text should be used as the primary source of textual content.
HOWEVER, for mathematical expressions and equations, the visual information from the image MUST be prioritized over the OCR text. OCR text for equations may be incomplete or incorrect and should only be used as a secondary reference.

The image should be used to resolve layout, structure, grouping, reading order, and ambiguous cases.

Each layout block should:
- group together relevant OCR text items based on spatial proximity and visual layout
- have a data-bbox that tightly encloses all OCR items belonging to that block
- preserve the correct reading order

Use the following labels:
- Caption
- Footnote
- Equation-Block
- List-Group
- Page-Header
- Page-Footer
- Image
- Section-Header
- Table
- Text
- Complex-Block
- Code-Block
- Form
- Table-Of-Contents
- Figure

Only use these tags ['math', 'br', 'i', 'b', 'u', 'del', 'sup', 'sub', 'table', 'tr', 'td', 'p', 'th', 'div', 'pre', 'h1', 'h2', 'h3', 'h4', 'h5', 'ul', 'ol', 'li', 'input', 'a', 'span', 'img', 'hr', 'tbody', 'small', 'caption', 'strong', 'thead', 'big', 'code'], and these attributes ['class', 'colspan', 'rowspan', 'display', 'checked', 'type', 'border', 'value', 'style', 'href', 'alt', 'align'].

Guidelines:
* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display mode for block math.
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property.
* Forms: Mark checkboxes and radio buttons properly.
* Text: join lines together properly into paragraphs using <p>...</p> tags. Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
* Use the simplest possible HTML structure that accurately represents the content of the block.
* Make sure the text is accurate and easy for a human to read and interpret. Reading order should be correct and natural.
""".strip()

# chandra ocr prompt without ocr info
prompt_user_original = """
OCR this image to HTML, arranged as layout blocks.  Each layout block should be a div with the data-bbox attribute representing the bounding box of the block in [x0, y0, x1, y1] format.  Bboxes are normalized 0-{bbox_scale}. The data-label attribute is the label for the block.

Use the following labels:
- Caption
- Footnote
- Equation-Block
- List-Group
- Page-Header
- Page-Footer
- Image
- Section-Header
- Table
- Text
- Complex-Block
- Code-Block
- Form
- Table-Of-Contents
- Figure

Only use these tags ['math', 'br', 'i', 'b', 'u', 'del', 'sup', 'sub', 'table', 'tr', 'td', 'p', 'th', 'div', 'pre', 'h1', 'h2', 'h3', 'h4', 'h5', 'ul', 'ol', 'li', 'input', 'a', 'span', 'img', 'hr', 'tbody', 'small', 'caption', 'strong', 'thead', 'big', 'code'], and these attributes ['class', 'colspan', 'rowspan', 'display', 'checked', 'type', 'border', 'value', 'style', 'href', 'alt', 'align'].

Guidelines:
* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property.
* Forms: Mark checkboxes and radio buttons properly.
* Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
* Use the simplest possible HTML structure that accurately represents the content of the block.
* Make sure the text is accurate and easy for a human to read and interpret.  Reading order should be correct and natural.
""".strip()
